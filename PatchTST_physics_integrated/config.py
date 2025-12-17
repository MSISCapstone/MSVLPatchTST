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
        self.target = 'T (degC)'  # Changed from 'OT' since that column was dropped
        self.freq = 't'  # minutely
        self.embed = 'timeF'

        # Forecasting task
        self.seq_len = 512    # Input sequence length (~3.5 days)
        self.label_len = 48   # Not used in PatchTST
        self.pred_len = 336   # Prediction length (~2.3 days)

        # Model parameters
        self.model = 'PhysicsIntegratedPatchTST'
        # After dropping the extra 'OT' column the dataset has 22 features
        # (21 weather variables + 2 hour-of-day features appended later)
        self.enc_in = 22      # Total input channels after dropping 'OT'
        self.dec_in = 22
        self.c_out = 22       # Output all channels
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
            'long_channel': {'patch_len': 24, 'stride': 12, 'weight': 0.5},  # 50% overlap for smooth long-term trends
            'short_channel': {'patch_len': 12, 'stride': 6, 'weight': 0.5}    # No overlap to capture rapid variations
        }

        # Hour-of-day feature configuration (indices adjusted for enc_in=22)
        self.hour_feature_indices = [20, 21]  # hour_sin, hour_cos

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
        # Define channel names based on the CSV header (OT removed). Hour features appended last.
        full_names = [
            'p (mbar)', 'T (degC)', 'Tpot (K)', 'Tdew (degC)', 'rh (%)', 'VPmax (mbar)', 'VPact (mbar)',
            'VPdef (mbar)', 'sh (g/kg)', 'H2OC (mmol/mol)', 'rho (g/m**3)', 'wv (m/s)', 'max. wv (m/s)',
            'wd (deg)', 'rain (mm)', 'raining (s)', 'SWDR (W/m2)', 'PAR (umol/m2/s)', 'max. PAR (umol/m2/s)',
            'Tlog (degC)', 'hour_sin', 'hour_cos'
        ]

        return {
            'long_channel': {
                'indices': list(range(22)),  # All input channels for enc_in=22
                'names': full_names,
                # Targets: rain (14), T (1), Tpot (2), wv (11)
                'output_indices': [14, 1, 2, 11],
                'description': 'Long channel predictors for rain, temperature, wind speed'
            },
            'short_channel': {
                'indices': list(range(22)),  # All input channels for enc_in=22
                'names': full_names,
                # Targets: raining (15), Tdew (3), max. wv (12)
                'output_indices': [15, 3, 12],
                'description': 'Short channel predictors for raining duration, dew point, max wind'
            }
        }
