class FullyConnectedStrategy(LayerStrategy):
    """Strategy for INT8 fully connected (linear/dense) layers"""
    
    def __init__(self):
        pass
    
    def get_layer_type(self):
        return "fully_connected"
    
    def generate_config(self):
        input_sizes = [128, 256, 512, 784, 1024]
        output_sizes = [10, 64, 128, 256, 512]
        
        config = {
            'input_size': np.random.choice(input_sizes),
            'output_size': np.random.choice(output_sizes),
            'use_bias': True,
            'data_range': (-128, 127),
            'weight_range': (-128, 127)
        }
        return config
    
    def generate_input_data(self, config):
        input_size = config['input_size']
        output_size = config['output_size']
        data_low, data_high = config['data_range']
        weight_low, weight_high = config['weight_range']
        
        # Ensure all generated data is explicitly int8
        input_data = np.random.randint(data_low, data_high + 1, size=input_size, dtype=np.int8)
        weights = np.random.randint(weight_low, weight_high + 1, size=(output_size, input_size), dtype=np.int8)
        
        if config['use_bias']:
            bias = np.random.randint(data_low, data_high + 1, size=output_size, dtype=np.int8)
        else:
            bias = np.zeros(output_size, dtype=np.int8)
        
        return {
            'input': input_data,
            'weights': weights,
            'bias': bias
        }
    
    def compute_golden(self, data_dict, config):
        """
        Golden model for INT8 Hardware.
        Note: Intermediate accumulation usually happens in INT32 to prevent overflow,
        but the final result is truncated/saturated back to INT8 for the DUT.
        """
        # Cast to int32 for the dot product calculation to avoid overflow mid-sum
        input_data = data_dict['input'].astype(np.int32)
        weights = data_dict['weights'].astype(np.int32)
        bias = data_dict['bias'].astype(np.int32)
        
        # Matrix multiply: (output_size, input_size) @ (input_size,)
        output = np.matmul(weights, input_data)
        
        if config['use_bias']:
            output = output + bias
        
        # --- INT8 SATURATION ---
        # Hardware typically clamps values that exceed 8-bit bounds
        output_int8 = np.clip(output, -128, 127)
        
        return output_int8.astype(np.int8)