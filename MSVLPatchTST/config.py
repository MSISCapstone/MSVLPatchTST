"""
Configuration for Physics-Integrated PatchTST with Predictor-Based Grouping
"""

class MSVLConfig:
    """
    Configuration class for Variable-Length PatchTST with Physics-Based Grouping
    and Hour-of-Day Integration
    """
    def __init__(self):
        self.random_seed = 42

        # Data parameters
        self.data = 'custom'
        self.root_path = './datasets/weather'
        self.data_path = 'weather_with_hour.csv'
        self.features = 'M'
        self.target = 'OT'  # Use 'OT' to match baseline
        self.freq = 't'  # minutely
        self.embed = 'timeF'

        # Forecasting task
        self.seq_len = 512    # Input sequence length (~3.5 days)
        self.label_len = 48   # Not used in PatchTST
        self.pred_len = 336   # Prediction length (~2.3 days)

        # Model parameters
        self.model = 'MSVLPatchTST'
        # Dataset has 22 features for MSVLPatchTST:
        # 20 weather variables (excluding OT) + hour_sin + hour_cos
        self.enc_in = 22      # Total input channels (without OT)
        self.dec_in = 22
        self.c_out = 6        # Output only 6 target weather channels
        self.d_model = 128
        self.n_heads = 16
        self.e_layers = 3
        self.d_layers = 1
        self.d_ff = 256
        self.dropout = 0.2
        self.fc_dropout = 0.2
        self.head_dropout = 0.0

        # Variable-length patching configuration (Predictor-Based grouping)
        self.channel_groups = self._define_channel_groups()
        self.patch_configs = {
            'short_channel': {'patch_len': 12, 'stride': 4, 'weight': 0.5},
            'long_channel': {'patch_len': 16, 'stride': 8, 'weight': 0.5}
        }
        
        # Max pooling for long channel preprocessing
        self.long_channel_pool_kernel = 4
        self.long_channel_pool_stride = 1

        # Hour-of-day feature configuration (indices adjusted for enc_in=22, OT excluded)
        self.hour_feature_indices = [20, 21]  # hour_sin, hour_cos (indices after excluding OT)

        # Encoder architecture
        self.use_cross_channel_encoder = False  # Set to True to use cross-channel embeddings
        self.use_cross_group_attention = True   # Cross-group attention (auto-disabled if cross-channel encoder is used)

        # Legacy PatchTST params (for compatibility)
        self.padding_patch = 'end'
        self.revin = 1  # Enable RevIN (applied only to weather channels)
        self.affine = 0
        self.subtract_last = 0
        self.decomposition = 0
        self.kernel_size = 25
        self.individual = 1

        # Training parameters
        self.batch_size = 64
        self.learning_rate = 0.0001
        self.train_epochs = 100
        self.patience = 10
        self.num_workers = 0
        self.lradj = 'type3'
        self.use_amp = False
        self.pct_start = 0.3

        # GPU
        self.use_gpu = True
        self.gpu = 0
        self.use_multi_gpu = False
        self.devices = '0'

        # Other
        self.checkpoints = './checkpoints_physics_integrated'
        self.output_attention = False
        self.embed_type = 0
        self.activation = 'gelu'
        self.distil = True
        self.factor = 1
        self.moving_avg = 25
        self.do_predict = False
        self.itr = 1
        self.des = 'MSVLExp'
        self.loss = 'mse'

    def _define_channel_groups(self):
        """Define long and short channel grouping with physics-based feature assignment"""
        # All input feature names (22 total)
        full_names = [
            'p (mbar)', 'T (degC)', 'Tpot (K)', 'Tdew (degC)', 'rh (%)', 'VPmax (mbar)', 'VPact (mbar)',
            'VPdef (mbar)', 'sh (g/kg)', 'H2OC (mmol/mol)', 'rho (g/m**3)', 'wv (m/s)', 'max. wv (m/s)',
            'wd (deg)', 'rain (mm)', 'raining (s)', 'SWDR (W/m2)', 'PAR (umol/m2/s)', 'max. PAR (umol/m2/s)',
            'Tlog (degC)', 'hour_sin', 'hour_cos'  # OT excluded
        ]
        
        # Long channel targets: p (mbar), T (degC), wv (m/s)
        # These have slower dynamics - use larger patch size
        long_target_indices = [0, 1, 11]  # p=0, T=1, wv=11
        long_target_names = ['p (mbar)', 'T (degC)', 'wv (m/s)']
        
        # Short channel targets: max. wv (m/s), rain (mm), raining (s)
        # These have faster dynamics - use smaller patch size
        short_target_indices = [12, 14, 15]  # max.wv=12, rain=14, raining=15
        short_target_names = ['max. wv (m/s)', 'rain (mm)', 'raining (s)']
        
        # Output indices in the final c_out=6 tensor:
        # [0, 1, 2] = long channel targets (p, T, wv)
        # [3, 4, 5] = short channel targets (max.wv, rain, raining)

        return {
            'long_channel': {
                'indices': list(range(22)),  # All 22 inputs as predictors
                'names': full_names,
                'target_indices': long_target_indices,  # Which input indices to predict
                'target_names': long_target_names,
                'output_indices': [0, 1, 2],  # Position in final output tensor
                'description': 'Long-scale channel for slow dynamics (p, T, wv)'
            },
            'short_channel': {
                'indices': list(range(22)),  # All 22 inputs as predictors
                'names': full_names,
                'target_indices': short_target_indices,  # Which input indices to predict
                'target_names': short_target_names,
                'output_indices': [3, 4, 5],  # Position in final output tensor
                'description': 'Short-scale channel for fast dynamics (max.wv, rain, raining)'
            }
        }
