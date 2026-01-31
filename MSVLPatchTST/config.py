"""
Configuration for Physics-Integrated PatchTST with Predictor-Based Grouping
"""

class PhysicsIntegratedConfig:
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
        self.model = 'PhysicsIntegratedPatchTST'
        # Dataset has 24 features total:
        # 20 weather variables + OT + hour_sin + hour_cos
        self.enc_in = 24      # Total input channels
        self.dec_in = 24
        self.c_out = 24       # Output all channels
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
            'short_channel': {'patch_len': 12, 'stride': 4, 'weight': 0.5},  # 50% overlap for smooth long-term trends
            'long_channel': {'patch_len': 16, 'stride': 8, 'weight': 0.5}    # No overlap to capture rapid variations
        }
        
        # Max pooling for long channel preprocessing
        self.long_channel_pool_kernel = 4
        self.long_channel_pool_stride = 1

        # Hour-of-day feature configuration (indices adjusted for enc_in=24)
        self.hour_feature_indices = [22, 23]  # hour_sin, hour_cos (OT is at index 21)

        # Encoder architecture
        self.use_cross_channel_encoder = False  # Set to True to use cross-channel embeddings
        self.use_cross_group_attention = True   # Cross-group attention (auto-disabled if cross-channel encoder is used)

        # Legacy PatchTST params (for compatibility)
        self.padding_patch = 'end'
        self.revin = 1
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
        self.des = 'PhysicsIntegratedExp'
        self.loss = 'mse'

    def _define_channel_groups(self):
        """Define long and short channel grouping"""
        # Define channel names based on the CSV header including OT and hour features
        full_names = [
            'p (mbar)', 'T (degC)', 'Tpot (K)', 'Tdew (degC)', 'rh (%)', 'VPmax (mbar)', 'VPact (mbar)',
            'VPdef (mbar)', 'sh (g/kg)', 'H2OC (mmol/mol)', 'rho (g/m**3)', 'wv (m/s)', 'max. wv (m/s)',
            'wd (deg)', 'rain (mm)', 'raining (s)', 'SWDR (W/m2)', 'PAR (umol/m2/s)', 'max. PAR (umol/m2/s)',
            'Tlog (degC)', 'OT', 'hour_sin', 'hour_cos'
        ]

        return {
            'short_channel': {
                'indices': list(range(24)),  # All input channels for enc_in=24
                'names': full_names,
                # Output all 24 features for MSE optimization across all variables
                'output_indices': list(range(24)),
                'description': 'Short channel predictors for all features'
            },
            'long_channel': {
                'indices': list(range(24)),  # All input channels for enc_in=24
                'names': full_names,
                # Output all 24 features for MSE optimization across all variables
                'output_indices': list(range(24)),
                'description': 'Long channel predictors for all features'
            }
        }
