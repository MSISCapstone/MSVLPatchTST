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
        self.c_out = 21       # Output only 21 weather channels (exclude hour features)
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
            'rain_predictors': {'patch_len': 24, 'stride': 12, 'weight': 0.25},
            'temperature_predictors': {'patch_len': 36, 'stride': 18, 'weight': 0.30},
            'wind_predictors': {'patch_len': 32, 'stride': 16, 'weight': 0.20},
            'other_variables': {'patch_len': 24, 'stride': 12, 'weight': 0.25}
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
        """Define predictor-based channel grouping"""
        return {
            'rain_predictors': {
                'indices': [10, 11, 3, 15, 13, 12, 5, 4, 6, 0, 2, 14, 7, 8, 9, 18, 21, 22],
                'names': ['rain (mm)', 'raining (s)', 'rh (%)', 'Tdew (degC)', 'H2OC (mmol/mol)', 
                          'sh (g/kg)', 'VPact (mbar)', 'VPmax (mbar)', 'VPdef (mbar)',
                          'p (mbar)', 'Tpot (K)', 'rho (g/m**3)', 
                          'wv (m/s)', 'max. wv (m/s)', 'wd (deg)', 'PAR (umol/m2/s)',
                          'hour_sin', 'hour_cos'],
                'output_indices': [10, 11],
                'description': 'Rain Predictors: self-history + humidity + vapor pressure + pressure + wind + radiation + hour'
            },
            'temperature_predictors': {
                'indices': [1, 2, 15, 18, 0, 14, 7, 8, 9, 3, 12, 13, 5, 4, 6, 21, 22],
                'names': ['T (degC)', 'Tpot (K)', 'Tdew (degC)', 'PAR (umol/m2/s)',
                          'p (mbar)', 'rho (g/m**3)', 
                          'wv (m/s)', 'max. wv (m/s)', 'wd (deg)',
                          'rh (%)', 'sh (g/kg)', 'H2OC (mmol/mol)', 
                          'VPact (mbar)', 'VPmax (mbar)', 'VPdef (mbar)',
                          'hour_sin', 'hour_cos'],
                'output_indices': [1, 2, 15],
                'description': 'Temperature Predictors: self-history + radiation + pressure + wind + humidity + vapor pressure + hour'
            },
            'wind_predictors': {
                'indices': [7, 8, 0, 2, 1, 15, 18, 3, 12, 13, 5, 4, 6, 14, 9, 21, 22],
                'names': ['wv (m/s)', 'max. wv (m/s)', 'p (mbar)', 'Tpot (K)',
                          'T (degC)', 'Tdew (degC)', 'PAR (umol/m2/s)',
                          'rh (%)', 'sh (g/kg)', 'H2OC (mmol/mol)', 
                          'VPact (mbar)', 'VPmax (mbar)', 'VPdef (mbar)',
                          'rho (g/m**3)', 'wd (deg)',
                          'hour_sin', 'hour_cos'],
                'output_indices': [7, 8],
                'description': 'Wind Speed Predictors: self-history + pressure + temperature + humidity + vapor pressure + density + wind direction + hour'
            },
            'other_variables': {
                'indices': [0, 3, 4, 5, 6, 9, 12, 13, 14, 16, 17, 18, 19, 20, 1, 2, 15, 7, 8, 21, 22],
                'names': ['p (mbar)', 'rh (%)', 'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)',
                          'wd (deg)', 'sh (g/kg)', 'H2OC (mmol/mol)', 'rho (g/m**3)',
                          'Tlog (degC)', 'CO2 (ppm)', 'PAR (umol/m2/s)', 'Tmax (degC)', 'Tmin (degC)',
                          'T (degC)', 'Tpot (K)', 'Tdew (degC)', 'wv (m/s)', 'max. wv (m/s)',
                          'hour_sin', 'hour_cos'],
                'output_indices': [0, 3, 4, 5, 6, 9, 12, 13, 14, 16, 17, 18, 19, 20],
                'description': 'Other Variables: pressure, humidity, vapor pressure, wind direction, moisture, density, derived temperature, CO2, radiation'
            }
        }
