"""
Fully-Connected (Dense) layer strategy for NVDLA nv_small (INT8).

FC layers are mapped onto the convolution engine as 1×1 direct convolutions:
    - Input:  flatten to (C, 1, 1) where C = input_features
    - Weight: (K, C, 1, 1)        where K = output_features
    - Output: (K, 1, 1)

The hardware pipeline is identical to convolution:
    CDMA → CSC → CMAC_A/B → CACC → SDP (passthrough) → DRAM

Golden-model pipeline:
    1. result[k] = sum(input[c] * weight[k][c] for c in 0..C-1)  — INT32
    2. result[k] = result[k] >> clip_truncate                     — CACC shift
    3. result[k] = clamp(result[k], -128, 127)                    — saturate INT8
"""

from strategy.LayerStrategy import LayerStrategy
from strategy.convolution_strategy import ConvolutionStrategy
import numpy as np


class FullyConnectedStrategy(LayerStrategy):
    """
    Strategy for NVDLA nv_small fully-connected (dense) layers.

    Internally delegates to ConvolutionStrategy since FC == 1×1 conv.
    Provides FC-friendly config keys (input_features, output_features)
    and converts them to the convolution parameter space.
    """

    def __init__(self):
        super().__init__()
        self._conv_strategy = ConvolutionStrategy()

    def get_layer_type(self):
        return "fully_connected"

    # ------------------------------------------------------------------ #
    #  Config translation: FC → Conv                                      #
    # ------------------------------------------------------------------ #
    @staticmethod
    def fc_to_conv_config(fc_config):
        """
        Translate FC-specific config keys into convolution config keys.

        FC config keys:
            input_features  : int   — number of input neurons (maps to num_channels)
            output_features : int   — number of output neurons (maps to num_kernels)
            data_range      : [min, max]
            weight_range    : [min, max]
            clip_truncate   : int

        Returns a dict compatible with ConvolutionStrategy / conv_configs.
        """
        input_features  = fc_config['input_features']
        output_features = fc_config['output_features']

        conv_config = {
            # Spatial: 1×1 (the entire input is "one pixel" with C channels)
            'input_shape':    [1, 1, input_features],
            'num_channels':   input_features,
            'num_kernels':    output_features,

            # 1×1 kernel — FC is a matrix multiply
            'kernel_h': 1,
            'kernel_w': 1,

            # No stride, padding, or dilation
            'stride_h': 1,
            'stride_w': 1,
            'padding_left':   0,
            'padding_right':  0,
            'padding_top':    0,
            'padding_bottom': 0,
            'padding_value':  0,
            'dilation_x': 1,
            'dilation_y': 1,

            # Data type
            'data_format': fc_config.get('data_format', 'INT8'),

            # Ranges
            'data_range':   fc_config.get('data_range', [-5, 5]),
            'weight_range': fc_config.get('weight_range', [-3, 3]),

            # CACC truncation
            'clip_truncate': fc_config.get('clip_truncate', 0),
        }
        return conv_config

    # ------------------------------------------------------------------ #
    #  Validation                                                         #
    # ------------------------------------------------------------------ #
    def validate_fc_config(self, config):
        """Raise ValueError when mandatory FC keys are missing."""
        required = ['input_features', 'output_features']
        for key in required:
            if key not in config:
                raise ValueError(f"Missing required FC config key: {key}")

        if config.get('data_format', 'INT8') != 'INT8':
            raise ValueError("nv_small FC supports INT8 only")

        if config['input_features'] < 1 or config['output_features'] < 1:
            raise ValueError("input_features and output_features must be >= 1")

    # ------------------------------------------------------------------ #
    #  Input-data generation                                              #
    # ------------------------------------------------------------------ #
    def generate_input_data(self, config):
        """
        Generate INT8 input vector as a (C, 1, 1) numpy array.

        This matches the convolution engine's expected CHW layout where
        H=W=1 and C = input_features.
        """
        conv_config = self.fc_to_conv_config(config)
        return self._conv_strategy.generate_input_data(conv_config)

    # ------------------------------------------------------------------ #
    #  Weight-data generation                                             #
    # ------------------------------------------------------------------ #
    def generate_weight_data(self, config):
        """
        Generate INT8 weight matrix as a (K, C, 1, 1) numpy array.

        K = output_features, C = input_features.
        """
        conv_config = self.fc_to_conv_config(config)
        return self._conv_strategy.generate_weight_data(conv_config)

    # ------------------------------------------------------------------ #
    #  NVDLA weight formatting                                            #
    # ------------------------------------------------------------------ #
    def format_weights_for_nvdla(self, weight_data):
        """Delegate to ConvolutionStrategy's NVDLA weight formatter."""
        return self._conv_strategy.format_weights_for_nvdla(weight_data)

    # ------------------------------------------------------------------ #
    #  Golden model                                                       #
    # ------------------------------------------------------------------ #
    def compute_golden(self, input_data, config, weight_data=None):
        """
        FC golden model: matrix multiply → truncate → saturate.

        Equivalent to 1×1 convolution golden model.

        Args:
            input_data  : np.int8 array (C, 1, 1)
            config      : FC config dict (with input_features, output_features)
            weight_data : np.int8 array (K, C, 1, 1)

        Returns:
            np.int8 array (K, 1, 1)
        """
        conv_config = self.fc_to_conv_config(config)
        return self._conv_strategy.compute_golden(input_data, conv_config, weight_data)
