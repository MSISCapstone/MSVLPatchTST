import torch
import torch.nn as nn
import numpy as np


class _ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""
    
    def __init__(self, dropout=0.0):
        """
        _ScaledDotProductAttention.__init__
        Purpose: Initializes the scaled dot-product attention module with dropout.
        Input: dropout (float): Dropout probability for attention weights.
        Output: None
        """
        super().__init__()
        self.attn_dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        """
        _ScaledDotProductAttention.forward
        Purpose: Computes scaled dot-product attention mechanism.
        Input: 
            query: [bs, seq_len, d_model] - Query tensor
            key: [bs, seq_len, d_model] - Key tensor  
            value: [bs, seq_len, d_model] - Value tensor
            attn_mask: [bs, seq_len, seq_len] optional - Attention mask
            key_padding_mask: [bs, seq_len] optional - Key padding mask
        Output: 
            output: [bs, seq_len, d_model] - Attention output
            attn_weights: [bs, seq_len, seq_len] - Attention weights
        """
        # query, key, value: [bs, seq_len, d_model]
        d_k = query.size(-1)
        
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k ** 0.5)
        
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e9)
        
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1), -1e9)
        
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        output = torch.matmul(attn_weights, value)
        
        return output, attn_weights


class CustomMultiheadAttention(nn.Module):
    """Custom multi-head attention with specified structure"""
    
    def __init__(self, d_model=128, n_heads=8, dropout=0.0):
        """
        CustomMultiheadAttention.__init__
        Purpose: Initializes custom multi-head attention module with Q/K/V projections and feed-forward.
        Input: 
            d_model (int): Model dimension
            n_heads (int): Number of attention heads
            dropout (float): Dropout probability
        Output: None
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # Query, Key, Value projections
        self.W_Q = nn.Linear(d_model, d_model, bias=True)
        self.W_K = nn.Linear(d_model, d_model, bias=True)
        self.W_V = nn.Linear(d_model, d_model, bias=True)
        
        # Scaled dot-product attention
        self.sdp_attn = _ScaledDotProductAttention(dropout=0.0)
        
        # Output projection
        self.to_out = nn.Sequential(
            nn.Linear(d_model, d_model * 4, bias=True),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model, bias=True),
        )
        
        # Additional dropout after output projection
        self.dropout_attn = nn.Dropout(dropout)
        
        # Layer normalization for attention output
        self.norm_attn = nn.LayerNorm(d_model)
    
    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        """
        CustomMultiheadAttention.forward
        Purpose: Performs multi-head attention with custom feed-forward and normalization.
        Input: 
            query: [bs, seq_len, d_model] - Query tensor
            key: [bs, seq_len, d_model] - Key tensor
            value: [bs, seq_len, d_model] - Value tensor
            attn_mask: [bs, seq_len, seq_len] optional - Attention mask
            key_padding_mask: [bs, seq_len] optional - Key padding mask
        Output: 
            output: [bs, seq_len, d_model] - Attention output
            attn_weights: [bs, seq_len, seq_len] - Attention weights
        """
        # Apply projections
        Q = self.W_Q(query)
        K = self.W_K(key)
        V = self.W_V(value)
        
        # Scaled dot-product attention
        attn_output, attn_weights = self.sdp_attn(Q, K, V, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        
        # Apply batch normalization
        attn_output = self.norm_attn(attn_output)
        
        # Output projection
        output = self.to_out(attn_output)
        
        # Additional dropout
        output = self.dropout_attn(output)
        
        return output, attn_weights


class Transpose(nn.Module):
    """Transpose module for sequential"""
    
    def __init__(self, dim0, dim1):
        """
        Transpose.__init__
        Purpose: Initializes transpose module to swap dimensions in tensor.
        Input: 
            dim0 (int): First dimension to transpose
            dim1 (int): Second dimension to transpose
        Output: None
        """
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1
    
    def forward(self, x):
        """
        Transpose.forward
        Purpose: Transposes specified dimensions of input tensor.
        Input: x: [bs, ...] - Input tensor with arbitrary dimensions
        Output: x: [bs, ...] - Transposed tensor with dim0 and dim1 swapped
        """
        return x.transpose(self.dim0, self.dim1)


class CrossChannelEncoder(nn.Module):
    """Encoder that creates embeddings across all channels simultaneously.
    
    Instead of processing each channel independently, this encoder:
    1. Patches the input across time for all channels
    2. Creates shared embeddings that capture cross-channel dependencies
    3. Uses attention to learn relationships between channels and time
    
    Benefits:
    - Better capture inter-variable dependencies
    - More parameter efficient
    - Can learn shared temporal patterns across variables
    """
    def __init__(self, n_input_channels, n_output_channels, context_window, target_window,
                 patch_len, stride, d_model=128, n_heads=8,
                 dropout=0.2, head_dropout=0.0, padding_patch='end'):
        """
        CrossChannelEncoder.__init__
        Purpose: Initializes cross-channel encoder that processes all channels together.
        Input:
            n_input_channels: Number of input channels
            n_output_channels: Number of output channels to predict
            context_window: Input sequence length
            target_window: Prediction sequence length
            patch_len: Length of each patch
            stride: Stride for patching
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout rate
            head_dropout: Dropout for attention heads
            padding_patch: Padding strategy ('end' or None)
        Output: None
        """
        super().__init__()
        self.n_input_channels = n_input_channels
        self.n_output_channels = n_output_channels
        self.context_window = context_window
        self.target_window = target_window
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.padding_patch = padding_patch
        
        # Calculate number of patches
        self.n_patches = (max(context_window, patch_len) - patch_len) // stride + 1
        if padding_patch == 'end':
            self.n_patches += 1
        
        # Patch embedding: projects [n_channels * patch_len] -> d_model
        self.patch_embedding = nn.Linear(n_input_channels * patch_len, d_model)
        
        # Positional encoding for patches
        self.pos_encoding = nn.Parameter(torch.randn(1, self.n_patches, d_model))
        
        # Channel encoding to help model distinguish channels
        self.channel_encoding = nn.Parameter(torch.randn(1, n_input_channels, d_model))
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            CustomMultiheadAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
            for _ in range(3)
        ])
        
        # Output projection: d_model -> n_output_channels * target_window
        self.head = nn.Linear(d_model * self.n_patches, n_output_channels * target_window)
        
        self.dropout = nn.Dropout(dropout)
        self.head_dropout = nn.Dropout(head_dropout)
        
    def forward(self, x, output_channel_mask):
        """
        CrossChannelEncoder.forward
        Purpose: Forward pass with cross-channel patching and attention.
        Input:
            x: [bs, n_input_channels, seq_len] - Input tensor
            output_channel_mask: List of bools indicating which channels to output
        Output:
            [bs, n_output_channels, target_window] - Predictions
        """
        bs, n_ch, seq_len = x.shape
        
        # Create patches: [bs, n_patches, n_channels * patch_len]
        patches = []
        for i in range(self.n_patches):
            start = i * self.stride
            end = start + self.patch_len
            if end > seq_len:
                if self.padding_patch == 'end':
                    # Pad with zeros
                    patch = torch.zeros(bs, n_ch, self.patch_len, device=x.device)
                    available = seq_len - start
                    if available > 0:
                        patch[:, :, :available] = x[:, :, start:seq_len]
                else:
                    break
            else:
                patch = x[:, :, start:end]
            
            # Flatten channels and time: [bs, n_channels * patch_len]
            patch = patch.reshape(bs, -1)
            patches.append(patch)
        
        # Stack patches: [bs, n_patches, n_channels * patch_len]
        x_patched = torch.stack(patches, dim=1)
        
        # Embed patches: [bs, n_patches, d_model]
        x_embed = self.patch_embedding(x_patched)
        
        # Add positional encoding
        x_embed = x_embed + self.pos_encoding
        x_embed = self.dropout(x_embed)
        
        # Apply transformer encoder layers
        for encoder_layer in self.encoder_layers:
            x_embed, _ = encoder_layer(x_embed, x_embed, x_embed)  # Unpack tuple (output, attn_weights)
        
        # Flatten patches: [bs, n_patches * d_model]
        x_flat = x_embed.reshape(bs, -1)
        
        # Project to output: [bs, n_output_channels * target_window]
        output = self.head(x_flat)
        output = self.head_dropout(output)
        
        # Reshape: [bs, n_output_channels, target_window]
        output = output.reshape(bs, self.n_output_channels, self.target_window)
        
        return output


class PerChannelEncoder(nn.Module):
    """Encoder for a group of channels with specific patch length.
    
    For groups with hour features integrated, the encoder learns
    the correlation between hour-of-day and physics variables directly.
    
    Uses custom multi-head attention per channel.
    """
    def __init__(self, n_input_channels, n_output_channels, context_window, target_window, 
                 patch_len, stride, d_model=128, n_heads=8, 
                 dropout=0.2, head_dropout=0.0, padding_patch='end'):
        """
        PerChannelEncoder.__init__
        Purpose: Initializes per-channel encoder for a group of weather channels with custom attention.
        Input: 
            n_input_channels (int): Number of input channels (including hour features)
            n_output_channels (int): Number of output channels (weather only)
            context_window (int): Input sequence length
            target_window (int): Prediction sequence length
            patch_len (int): Length of each patch
            stride (int): Stride for patch creation
            d_model (int): Model dimension
            n_heads (int): Number of attention heads
            dropout (float): Dropout probability
            head_dropout (float): Head dropout probability
            padding_patch (str): Padding strategy ('end')
        Output: None
        """
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
        
        # Custom multi-head attention for each channel
        self.attentions = nn.ModuleList([
            CustomMultiheadAttention(d_model, n_heads, dropout) 
            for _ in range(n_input_channels)
        ])
        
        # Per-channel prediction heads (only for OUTPUT channels, not hour features)
        self.head_nf = d_model * patch_num
        self.final_head = nn.Sequential(
            nn.Flatten(start_dim=-2, end_dim=-1),
            nn.Linear(self.head_nf, target_window),
            nn.Dropout(0.0)
        )
        
    def forward(self, x, output_channel_mask):
        """
        PerChannelEncoder.forward
        Purpose: Encodes input channels using per-channel attention and generates predictions for output channels.
        Input: 
            x: [bs, n_input_channels, seq_len] - Input tensor with all channels for this group
            output_channel_mask: [n_input_channels] bool list - True for channels to output predictions
        Output: 
            outputs: [bs, n_output_channels, pred_len] - Predictions for output channels only
        """
        bs, n_ch, seq_len = x.shape
        
        # Padding if needed
        if self.padding_patch == 'end':
            x = self.padding_layer(x)
        
        # Create patches: [bs, n_channels, patch_num, patch_len]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        
        # Process each channel independently
        outputs = []
        
        for ch in range(n_ch):
            ch_x = x[:, ch, :, :]  # [bs, patch_num, patch_len]
            
            # Project patches to d_model
            ch_x = self.W_P(ch_x)  # [bs, patch_num, d_model]
            
            # Add positional encoding
            ch_x = ch_x + self.W_pos
            
            # Custom multi-head attention (per channel)
            ch_z, _ = self.attentions[ch](ch_x, ch_x, ch_x)  # [bs, patch_num, d_model]
            
            # Only create output for weather channels, not hour features
            if output_channel_mask[ch]:
                ch_out = self.final_head(ch_z)  # [bs, target_window]
                outputs.append(ch_out)
        
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
        """
        CrossGroupAttention.__init__
        Purpose: Initializes cross-group attention module for fusing long and short channel predictions.
        Input: 
            n_channels (int): Number of channels to process
            d_model (int): Model dimension
            n_heads (int): Number of attention heads
            dropout (float): Dropout probability
        Output: None
        """
        super().__init__()
        self.n_channels = n_channels
        self.d_model = d_model
    
        # Project each channel's prediction to d_model
        self.channel_proj = nn.Linear(1, d_model)
        # Layer normalization for input
        self.norm1 = nn.LayerNorm(d_model)
        
        # Cross-channel attention
        self.cross_attn = CustomMultiheadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout
        )
        
        # Feed-forward network for refinement (2 layers)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.norm_ffn = nn.LayerNorm(d_model)
        
        # Project back to prediction
        self.output_proj = nn.Linear(d_model, 1)
        
    def forward(self, x):
        """
        CrossGroupAttention.forward
        Purpose: Applies cross-group attention to fuse predictions from different channel groups.
        Input: x: [bs, pred_len, n_channels] - Independent group predictions
        Output: x: [bs, pred_len, n_channels] - Cross-group refined predictions
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
        x_ffn = self.ffn(self.norm_ffn(x_proj))
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
        """
        PhysicsIntegratedPatchTST.__init__
        Purpose: Initializes the complete physics-integrated PatchTST model with dual encoders and cross-group attention.
        Input: configs - Configuration object with model parameters
        Output: None
        """
        super().__init__()
        
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in  # 23 (21 weather + 2 hour)
        self.channel_groups = configs.channel_groups
        self.patch_configs = configs.patch_configs
        self.c_out = configs.c_out  # Output all 23 channels
        self.hour_indices = set(configs.hour_feature_indices)  # {21, 22}
        self.use_cross_channel = configs.use_cross_channel_encoder  # New config option
        
        # Cross-group attention: disable if using cross-channel encoder (already captures dependencies)
        self.use_cross_group_attn = configs.use_cross_group_attention and not self.use_cross_channel
        
        # RevIN for normalization (only for weather channels, not hour features)
        self.revin = configs.revin
        if self.revin:
            from MSVLPatchTST.layers.RevIN import RevIN
            # RevIN only for c_out channels (weather features), not hour features
            self.revin_layer = RevIN(configs.c_out, affine=configs.affine, 
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
            
            # Choose encoder based on configuration
            if self.use_cross_channel:
                self.encoders[group_name] = CrossChannelEncoder(
                    n_input_channels=len(indices),
                    n_output_channels=len(actual_output_indices),
                    context_window=configs.seq_len,
                    target_window=configs.pred_len,
                    patch_len=patch_config['patch_len'],
                    stride=patch_config['stride'],
                    d_model=configs.d_model,
                    n_heads=configs.n_heads,
                    dropout=configs.dropout,
                    head_dropout=configs.head_dropout,
                    padding_patch=configs.padding_patch
                )
            else:
                self.encoders[group_name] = PerChannelEncoder(
                    n_input_channels=len(indices),
                    n_output_channels=len(actual_output_indices),
                    context_window=configs.seq_len,
                    target_window=configs.pred_len,
                    patch_len=patch_config['patch_len'],
                    stride=patch_config['stride'],
                    d_model=configs.d_model,
                    n_heads=configs.n_heads,
                    dropout=configs.dropout,
                    head_dropout=configs.head_dropout,
                    padding_patch=configs.padding_patch
                )
                
        # Collect all target indices in order
        self.target_indices = []
        for group_name in self.channel_groups.keys():
            self.target_indices.extend(self.group_info[group_name]['output_indices'])
        
        # Cross-group attention layer (only if enabled)
        if self.use_cross_group_attn:
            self.cross_group_attn = CrossGroupAttention(
                n_channels=self.c_out,  # Dynamic
                d_model=configs.d_model // 2,
                n_heads=4,
                dropout=configs.dropout
            )
            # Learnable mixing weight
            self.cross_group_weight = nn.Parameter(torch.tensor(0.3))
        
    def forward(self, x):
        """
        PhysicsIntegratedPatchTST.forward
        Purpose: Performs forward pass through the physics-integrated PatchTST model with group-specific encoding and cross-group attention.
        Input: x - [bs, seq_len, 23] tensor with 21 weather channels + 2 hour features
        Output: output - [bs, pred_len, 21] tensor with predictions for all 21 weather channels
        """
        bs = x.shape[0]
        
        # Permute to [bs, enc_in, seq_len]
        x = x.permute(0, 2, 1)
        
        # Apply RevIN normalization only to weather channels (first c_out channels)
        if self.revin:
            weather_x = x[:, :self.c_out, :]  # [bs, c_out, seq_len]
            weather_x = weather_x.permute(0, 2, 1)  # [bs, seq_len, c_out]
            weather_x = self.revin_layer(weather_x, 'norm')  # Normalize
            weather_x = weather_x.permute(0, 2, 1)  # [bs, c_out, seq_len]
            x[:, :self.c_out, :] = weather_x  # Put back normalized weather channels
        
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
        
        # Step 2: Cross-group refinement (only if enabled)
        if self.use_cross_group_attn:
            cross_output = self.cross_group_attn(output)
            # Blend: original group-specific + cross-group refined
            alpha = torch.sigmoid(self.cross_group_weight)
            output = (1 - alpha) * output + alpha * cross_output
        
        # Step 3: RevIN denormalization (only for weather channels)
        if self.revin:
            output = self.revin_layer(output, 'denorm')
        
        return output
