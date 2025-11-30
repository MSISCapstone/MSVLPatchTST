__all__ = ['PatchTST']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from layers.PatchTST_backbone import PatchTST_backbone
from layers.PatchTST_layers import series_decomp


class MultiScalePatchTST(nn.Module):
    """Multi-scale PatchTST with variable-length patches.
    
    Creates multiple PatchTST backbones with different patch lengths and fuses their outputs.
    Each scale captures different temporal patterns (e.g., short patches for rapid changes,
    long patches for slow trends/diurnal cycles).
    """
    def __init__(self, c_in, context_window, target_window, patch_lengths, patch_strides, 
                 patch_weights, max_seq_len=1024, n_layers=3, d_model=128, n_heads=16, 
                 d_k=None, d_v=None, d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., 
                 act="gelu", key_padding_mask='auto', padding_var=None, attn_mask=None, 
                 res_attention=True, pre_norm=False, store_attn=False, pe='zeros', 
                 learn_pe=True, fc_dropout=0., head_dropout=0, padding_patch=None, 
                 pretrain_head=False, head_type='flatten', individual=False, revin=True, 
                 affine=True, subtract_last=False, channel_independent=True, verbose=False, **kwargs):
        super().__init__()
        
        self.patch_lengths = patch_lengths
        self.patch_strides = patch_strides
        self.patch_weights = patch_weights
        self.n_scales = len(patch_lengths)
        
        # Create separate encoder for each scale
        self.encoders = nn.ModuleList()
        for patch_len, stride in zip(patch_lengths, patch_strides):
            encoder = PatchTST_backbone(
                c_in=c_in, 
                context_window=context_window,
                target_window=target_window,
                patch_len=patch_len, 
                stride=stride,
                max_seq_len=max_seq_len,
                n_layers=n_layers,
                d_model=d_model,
                n_heads=n_heads,
                d_k=d_k,
                d_v=d_v,
                d_ff=d_ff,
                norm=norm,
                attn_dropout=attn_dropout,
                dropout=dropout,
                act=act,
                key_padding_mask=key_padding_mask,
                padding_var=padding_var,
                attn_mask=attn_mask,
                res_attention=res_attention,
                pre_norm=pre_norm,
                store_attn=store_attn,
                pe=pe,
                learn_pe=learn_pe,
                fc_dropout=fc_dropout,
                head_dropout=head_dropout,
                padding_patch=padding_patch,
                pretrain_head=pretrain_head,
                head_type=head_type,
                individual=individual,
                revin=revin,
                affine=affine,
                subtract_last=subtract_last,
                channel_independent=channel_independent,
                verbose=verbose,
                **kwargs
            )
            self.encoders.append(encoder)
        
        if verbose:
            print(f"MultiScalePatchTST initialized with {self.n_scales} scales:")
            for i, (pl, st, w) in enumerate(zip(patch_lengths, patch_strides, patch_weights)):
                num_patches = int((context_window - pl) / st + 1)
                print(f"  Scale {i+1}: patch_len={pl}, stride={st}, weight={w:.3f}, patches={num_patches}")
    
    def forward(self, x):
        # x: [bs x nvars x seq_len]
        multi_scale_outputs = []
        
        for encoder, weight in zip(self.encoders, self.patch_weights):
            output = encoder(x)  # [bs x nvars x target_window]
            multi_scale_outputs.append(output * weight)
        
        # Weighted sum of all scales
        # Each scale is already weighted, so just sum
        fused_output = torch.stack(multi_scale_outputs, dim=0).sum(dim=0)
        # fused_output: [bs x nvars x target_window]
        
        return fused_output


class Model(nn.Module):
    def __init__(self, configs, max_seq_len:Optional[int]=1024, d_k:Optional[int]=None, d_v:Optional[int]=None, norm:str='BatchNorm', attn_dropout:float=0., 
                 act:str="gelu", key_padding_mask:bool='auto',padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, 
                 pre_norm:bool=False, store_attn:bool=False, pe:str='zeros', learn_pe:bool=True, pretrain_head:bool=False, head_type = 'flatten', verbose:bool=False, **kwargs):
        
        super().__init__()
        
        # load parameters
        c_in = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.pred_len
        
        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout
        
        individual = configs.individual
    
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch
        
        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last
        
        decomposition = configs.decomposition
        kernel_size = configs.kernel_size
        
        # NEW: Channel interaction mode
        channel_independent = getattr(configs, 'channel_independent', True)
        
        # NEW: Multi-scale patching mode
        multi_scale = getattr(configs, 'multi_scale', 0)
        
        # model
        self.decomposition = decomposition
        self.multi_scale = multi_scale
        
        if self.multi_scale:
            # Multi-scale mode: use variable-length patches
            patch_lengths = getattr(configs, 'patch_lengths', [patch_len])
            patch_strides = getattr(configs, 'patch_strides', [stride])
            patch_weights = getattr(configs, 'patch_weights', [1.0])
            
            if verbose:
                print(f"Using Multi-Scale PatchTST with {len(patch_lengths)} scales")
            
            if self.decomposition:
                self.decomp_module = series_decomp(kernel_size)
                self.model_trend = MultiScalePatchTST(
                    c_in=c_in, context_window=context_window, target_window=target_window,
                    patch_lengths=patch_lengths, patch_strides=patch_strides, patch_weights=patch_weights,
                    max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                    n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                    dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                    attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                    pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch=padding_patch,
                    pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                    subtract_last=subtract_last, channel_independent=channel_independent, verbose=verbose, **kwargs)
                self.model_res = MultiScalePatchTST(
                    c_in=c_in, context_window=context_window, target_window=target_window,
                    patch_lengths=patch_lengths, patch_strides=patch_strides, patch_weights=patch_weights,
                    max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                    n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                    dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                    attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                    pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch=padding_patch,
                    pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                    subtract_last=subtract_last, channel_independent=channel_independent, verbose=verbose, **kwargs)
            else:
                self.model = MultiScalePatchTST(
                    c_in=c_in, context_window=context_window, target_window=target_window,
                    patch_lengths=patch_lengths, patch_strides=patch_strides, patch_weights=patch_weights,
                    max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                    n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                    dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                    attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                    pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch=padding_patch,
                    pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                    subtract_last=subtract_last, channel_independent=channel_independent, verbose=verbose, **kwargs)
        else:
            # Single-scale mode: use standard fixed-length patches
            if verbose:
                print(f"Using Single-Scale PatchTST with patch_len={patch_len}, stride={stride}")
            
            if self.decomposition:
                self.decomp_module = series_decomp(kernel_size)
                self.model_trend = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                      max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                      n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                      dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                      attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                      pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                      pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                      subtract_last=subtract_last, channel_independent=channel_independent, verbose=verbose, **kwargs)
                self.model_res = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                      max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                      n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                      dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                      attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                      pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                      pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                      subtract_last=subtract_last, channel_independent=channel_independent, verbose=verbose, **kwargs)
            else:
                self.model = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                      max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                      n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                      dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                      attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                      pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                      pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                      subtract_last=subtract_last, channel_independent=channel_independent, verbose=verbose, **kwargs)
    
    
    def forward(self, x):           # x: [Batch, Input length, Channel]
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res_init, trend_init = res_init.permute(0,2,1), trend_init.permute(0,2,1)  # x: [Batch, Channel, Input length]
            res = self.model_res(res_init)
            trend = self.model_trend(trend_init)
            x = res + trend
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        else:
            x = x.permute(0,2,1)    # x: [Batch, Channel, Input length]
            x = self.model(x)
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        return x