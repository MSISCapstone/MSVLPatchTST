"""
Model architectures for Physics-Integrated PatchTST
"""

import torch
import torch.nn as nn
import numpy as np


class PerChannelEncoder(nn.Module):
    """Encoder for a group of channels with specific patch length.
    
    For groups with hour features integrated, the encoder learns
    the correlation between hour-of-day and physics variables directly.
    
    Uses a 2-layer feedforward network for learning patterns.
    """
    def __init__(self, n_input_channels, n_output_channels, context_window, target_window, 
                 patch_len, stride, n_layers=3, d_model=128, n_heads=8, d_ff=256, 
                 dropout=0.2, head_dropout=0.0, padding_patch='end'):
        super().__init__()
        
        self.n_input_channels = n_input_channels   # Includes hour features if integrated
        self.n_output_channels = n_output_channels  # Only weather channels
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.d_model = d_model
        
        # Calculate patch count
        patch_num = int((context_window - patch_len) / stride + 1)
        if padding_patch == 'end':
            self.padding_layer = nn.ReplicationPad1d((0, stride))
            patch_num += 1
        self.patch_num = patch_num
        
        # Patch embedding (project patch_len → d_model)
        self.W_P = nn.Linear(patch_len, d_model)
        
        # Positional encoding (learnable)
        self.W_pos = nn.Parameter(torch.zeros(1, patch_num, d_model))
        nn.init.normal_(self.W_pos, std=0.02)
        
        # Feedforward Network (2 layers)
        self.deep_ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Per-channel prediction heads (only for OUTPUT channels, not hour features)
        self.head_nf = d_model * patch_num
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Flatten(start_dim=-2),
                nn.Linear(self.head_nf, target_window),
                nn.Dropout(head_dropout)
            ) for _ in range(n_output_channels)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, output_channel_mask):
        """
        Args:
            x: [bs, n_input_channels, seq_len] - includes hour features if integrated
            output_channel_mask: list of bools, True for channels to output
        Returns:
            outputs: [bs, n_output_channels, pred_len] - only weather channels
        """
        bs, n_ch, seq_len = x.shape
        
        # Padding if needed
        if self.padding_patch == 'end':
            x = self.padding_layer(x)
        
        # Create patches: [bs, n_channels, patch_num, patch_len]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        
        # Process each channel independently
        outputs = []
        head_idx = 0
        
        for ch in range(n_ch):
            ch_x = x[:, ch, :, :]  # [bs, patch_num, patch_len]
            
            # Project patches to d_model
            ch_x = self.W_P(ch_x)  # [bs, patch_num, d_model]
            
            # Add positional encoding
            ch_x = self.dropout(ch_x + self.W_pos)
            
            # Apply feedforward for pattern learning
            ch_x = ch_x + self.deep_ffn(self.ffn_norm(ch_x))  # Residual + FFN
            
            # Transformer encoder (channel-independent)
            ch_z = self.encoder(ch_x)  # [bs, patch_num, d_model]
            
            # Only create output for weather channels, not hour features
            if output_channel_mask[ch]:
                ch_out = self.heads[head_idx](ch_z)  # [bs, target_window]
                outputs.append(ch_out)
                head_idx += 1
        
        # Stack outputs: [bs, n_output_channels, target_window]
        output = torch.stack(outputs, dim=1)
        return output


class CrossGroupAttention(nn.Module):
    """
    Cross-Group Attention Layer to learn inter-variable dependencies.
    
    This allows the model to learn physical couplings between groups:
    - Temperature -> Humidity (warm air holds more moisture)
    - Humidity -> Precipitation (saturation triggers rain)
    - Pressure gradients -> Wind
    - Temperature -> Convection -> Rain
    """
    def __init__(self, n_channels, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.n_channels = n_channels
        self.d_model = d_model
    
        # Project each channel's prediction to d_model
        self.channel_proj = nn.Linear(1, d_model)
        # Layer normalization for input
        self.norm1 = nn.LayerNorm(d_model)
        
        # Cross-channel attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network for refinement (2 layers)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)
        
        # Project back to prediction
        self.output_proj = nn.Linear(d_model, 1)
        
    def forward(self, x):
        """
        Args:
            x: [bs, pred_len, n_channels] - independent group predictions
        Returns:
            x: [bs, pred_len, n_channels] - cross-group refined predictions
        """
        bs, pred_len, n_ch = x.shape
        
        # Reshape: [bs * pred_len, n_channels, 1]
        x_reshaped = x.reshape(bs * pred_len, n_ch, 1)
        
        # Project to d_model: [bs * pred_len, n_channels, d_model]
        x_proj = self.channel_proj(x_reshaped)
        
        # Pre-norm for attention
        x_norm = self.norm1(x_proj)
        
        # Cross-channel attention (channels attend to each other)
        x_attn, _ = self.cross_attn(x_norm, x_norm, x_norm)
        x_proj = x_proj + x_attn  # Residual connection
        
        # Feed-forward refinement
        x_ffn = self.ffn(self.norm2(x_proj))
        x_proj = x_proj + x_ffn  # Residual connection
        
        # Project back: [bs * pred_len, n_channels, 1]
        x_out = self.output_proj(x_proj)
        
        # Reshape: [bs, pred_len, n_channels]
        x_out = x_out.reshape(bs, pred_len, n_ch)
        
        return x_out


class PhysicsIntegratedPatchTST(nn.Module):
    """
    Predictor-based PatchTST with hour features and Cross-Group Attention.
    
    Architecture:
    1. Group-specific encoders for prediction targets (Rain, Temperature, Wind)
    2. Each group uses relevant predictor variables
    3. Cross-group attention learns inter-variable dependencies
    4. Output predictions for all 21 weather channels
    
    Key Features:
    - Hour features (hour_sin, hour_cos) are included in all predictor groups
    - Each encoder learns correlations between predictors and target
    - Cross-group attention enables learning couplings between variables
    - Only weather channels are predicted (hour features are input-only)
    """
    def __init__(self, configs):
        super().__init__()
        
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in  # 23 (21 weather + 2 hour)
        self.c_out = configs.c_out    # 21 (weather only)
        self.channel_groups = configs.channel_groups
        self.patch_configs = configs.patch_configs
        self.hour_indices = set(configs.hour_feature_indices)  # {21, 22}
        
        # RevIN for normalization (all input channels)
        self.revin = configs.revin
        if self.revin:
            from layers.RevIN import RevIN
            self.revin_layer = RevIN(configs.enc_in, affine=configs.affine, 
                                      subtract_last=configs.subtract_last)
        
        # Create encoder for each channel group
        self.encoders = nn.ModuleDict()
        self.group_info = {}
        
        for group_name, group_info in self.channel_groups.items():
            indices = group_info['indices']
            output_indices = group_info.get('output_indices', [])
            patch_config = self.patch_configs[group_name]
            
            # Separate weather channels from hour channels in this group
            weather_indices = [i for i in indices if i not in self.hour_indices]
            hour_indices_in_group = [i for i in indices if i in self.hour_indices]
            
            # If output_indices is specified, only those channels are outputs
            if output_indices:
                actual_output_indices = output_indices
            else:
                actual_output_indices = weather_indices
            
            # Create mask: True for channels that should be output
            output_mask = [i in actual_output_indices for i in indices]
            
            self.group_info[group_name] = {
                'all_indices': indices,
                'weather_indices': weather_indices,
                'hour_indices': hour_indices_in_group,
                'output_indices': actual_output_indices,
                'output_mask': output_mask,
                'n_input': len(indices),
                'n_output': len(actual_output_indices)
            }
            
            self.encoders[group_name] = PerChannelEncoder(
                n_input_channels=len(indices),
                n_output_channels=len(actual_output_indices),
                context_window=configs.seq_len,
                target_window=configs.pred_len,
                patch_len=patch_config['patch_len'],
                stride=patch_config['stride'],
                n_layers=configs.e_layers,
                d_model=configs.d_model,
                n_heads=configs.n_heads,
                d_ff=configs.d_ff,
                dropout=configs.dropout,
                head_dropout=configs.head_dropout,
                padding_patch=configs.padding_patch
            )
        
        self.group_weights = {name: cfg['weight'] for name, cfg in self.patch_configs.items()}
        
        # Cross-group attention layer
        self.cross_group_attn = CrossGroupAttention(
            n_channels=configs.c_out,  # 21 weather channels
            d_model=configs.d_model // 2,
            n_heads=4,
            dropout=configs.dropout
        )
        
        # Learnable mixing weight
        self.cross_group_weight = nn.Parameter(torch.tensor(0.3))
        
    def forward(self, x):
        """
        Args:
            x: [bs, seq_len, 23] - 21 weather + 2 hour features
        Returns:
            output: [bs, pred_len, 21] - only weather predictions
        """
        bs = x.shape[0]
        
        # Permute to [bs, enc_in, seq_len]
        x = x.permute(0, 2, 1)
        
        # Apply RevIN normalization
        if self.revin:
            x = x.permute(0, 2, 1)
            x = self.revin_layer(x, 'norm')
            x = x.permute(0, 2, 1)
        
        # Step 1: Group-independent encoding
        all_outputs = torch.zeros(bs, self.c_out, self.pred_len, device=x.device)
        
        for group_name, encoder in self.encoders.items():
            info = self.group_info[group_name]
            
            # Extract all channels for this group (including hour features)
            group_x = x[:, info['all_indices'], :]
            
            # Encode - returns only specified output channels
            group_out = encoder(group_x, info['output_mask'])
            
            # Place outputs in correct positions
            for i, ch_idx in enumerate(info['output_indices']):
                all_outputs[:, ch_idx, :] = group_out[:, i, :]
        
        # Permute to [bs, pred_len, c_out]
        output = all_outputs.permute(0, 2, 1)
        
        # Step 2: Cross-group refinement
        cross_output = self.cross_group_attn(output)
        
        # Blend: original group-specific + cross-group refined
        alpha = torch.sigmoid(self.cross_group_weight)
        output = (1 - alpha) * output + alpha * cross_output
        
        # Step 3: RevIN denormalization
        if self.revin:
            temp_output = torch.zeros(bs, self.pred_len, self.enc_in, device=x.device)
            temp_output[:, :, :self.c_out] = output
            temp_output = self.revin_layer(temp_output, 'denorm')
            output = temp_output[:, :, :self.c_out]
        
        return output
