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
        self.target = 'OT'
        self.freq = 't'  # minutely
        self.embed = 'timeF'

        # Forecasting task
        self.seq_len = 512    # Input sequence length (~3.5 days)
        self.label_len = 48   # Not used in PatchTST
        self.pred_len = 336   # Prediction length (~2.3 days)

        # Model parameters
        self.model = 'PhysicsIntegratedPatchTST'
        self.enc_in = 23      # Total channels: 21 weather + 2 hour-of-day
        self.dec_in = 23
        self.c_out = 23       # Output all 23 channels (optimize only on targets)
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
            'long_channel': {'patch_len': 24, 'stride': 12, 'weight': 0.5},
            'short_channel': {'patch_len': 6, 'stride': 3, 'weight': 0.5}
        }

        # Hour-of-day feature configuration
        self.hour_feature_indices = [21, 22]  # hour_sin, hour_cos

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
        # Define all 23 channel names (21 weather + 2 hour features)
        full_names = ['p (mbar)', 'T (degC)', 'Tpot (K)', 'rh (%)', 'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)', 
                      'wv (m/s)', 'max. wv (m/s)', 'wd (deg)', 'rain (mm)', 'raining (s)', 'sh (g/kg)', 
                      'H2OC (mmol/mol)', 'rho (g/m**3)', 'Tdew (degC)', 'Tlog (degC)', 'CO2 (ppm)', 
                      'PAR (umol/m2/s)', 'Tmax (degC)', 'Tmin (degC)', 'hour_sin', 'hour_cos']
        
        return {
            'long_channel': {
                'indices': list(range(23)),  # All 23 input channels
                'names': full_names,
                'output_indices': [10, 1, 2, 7],  # rain (mm), T (degC), Tpot (K), wv (m/s) - targets for optimization
                'description': 'Long channel predictors for rain, temperature, wind speed'
            },
            'short_channel': {
                'indices': list(range(23)),  # All 23 input channels
                'names': full_names,
                'output_indices': [11, 15, 8],  # raining (s), Tdew (degC), max. wv (m/s) - targets for optimization
                'description': 'Short channel predictors for raining duration, dew point, max wind'
            }
        }
