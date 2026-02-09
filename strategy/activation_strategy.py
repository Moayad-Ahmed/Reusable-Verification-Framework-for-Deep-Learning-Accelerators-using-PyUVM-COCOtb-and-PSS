from strategy.LayerStrategy import LayerStrategy
import numpy as np

class ActivationStrategy(LayerStrategy):
    """Strategy for activation layers using 8-bit signed integers"""
    
    def __init__(self):
        self.activation_types = ['relu', 'sigmoid', 'tanh', 'softmax']
        # 8-bit signed integer constants
        self.DATA_WIDTH = 8
        self.MIN_VAL = -128
        self.MAX_VAL = 127
        
    def get_layer_type(self):
        return "activation"
    
    def float_to_int8(self, value):
        """Convert floating point to 8-bit signed integer"""
        int_val = int(round(value))
        # Clamp to 8-bit signed range [-128, 127]
        int_val = max(self.MIN_VAL, min(self.MAX_VAL, int_val))
        return int_val
    
    def int8_to_float(self, int_val):
        """Convert 8-bit signed integer to floating point"""
        return float(int_val)
    
    def generate_input_data(self, config):
        """
        Generate input data for activation layers
        
        Example config:
        {
            'activation_type': 'relu',  # 'relu', 'sigmoid', 'tanh', 'softmax'
            'input_shape': (4, 4),      # Shape of input
            'data_range': (-128, 127)   # Range for 8-bit signed integers
        }
        """
        h, w = config.get('input_shape', (4, 4))
        low, high = config.get('data_range', (-128, 127))
        
        # Generate random input as 8-bit signed integers
        input_data = np.random.randint(low, high + 1, size=(h, w), dtype=np.int8)
            
        return input_data
    
    def compute_golden(self, input_data, config):
        """
        Golden model that EXACTLY matches hardware approximations.
        Hardware uses piecewise linear approximations, not real activation functions.
        
        Supports: ReLU, Sigmoid (approx), Tanh (approx), Softmax (simplified)
        """
        activation_type = config['activation_type'].lower()
        
        # Ensure input is 8-bit signed integers
        input_working = np.clip(input_data, self.MIN_VAL, self.MAX_VAL).astype(np.int8)
        
        output = np.zeros_like(input_working, dtype=np.int8)
        
        if activation_type == 'relu':
            # ReLU: f(x) = max(0, x)
            # Hardware: if (x < 0) then 0 else x
            output = np.where(input_working < 0, 0, input_working).astype(np.int8)
            
        elif activation_type == 'sigmoid':
            # Sigmoid Approximation (matches hardware piecewise):
            # f(x) = -96   if x < -64
            # f(x) = -32   if -64 <= x < 0
            # f(x) = 32    if 0 <= x < 64
            # f(x) = 96    if x >= 64
            output = np.select(
                [input_working < -64, input_working < 0, input_working < 64, input_working >= 64],
                [-96, -32, 32, 96],
                default=96
            ).astype(np.int8)
            
        elif activation_type == 'tanh':
            # Tanh Approximation (matches hardware piecewise):
            # f(x) = -112  if x < -64
            # f(x) = -48   if -64 <= x < 0
            # f(x) = 48    if 0 <= x < 64
            # f(x) = 112   if x >= 64
            output = np.select(
                [input_working < -64, input_working < 0, input_working < 64, input_working >= 64],
                [-112, -48, 48, 112],
                default=112
            ).astype(np.int8)
            
        elif activation_type == 'softmax':
            # Simplified Softmax (matches hardware):
            # Find max value, then subtract max from each element
            # This is a numerical stability trick: softmax(x - max(x)) = softmax(x)
            max_val = np.max(input_working)
            # Use int16 to avoid overflow during subtraction, then clamp
            output = input_working.astype(np.int16) - int(max_val)
            # Clamp to signed 8-bit range
            output = np.clip(output, self.MIN_VAL, self.MAX_VAL).astype(np.int8)
        
        else:
            raise ValueError(f"Unsupported activation type: {activation_type}")
        
        return output
    