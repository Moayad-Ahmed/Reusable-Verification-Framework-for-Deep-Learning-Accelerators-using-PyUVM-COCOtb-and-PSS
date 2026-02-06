import numpy as np
from LayerStrategy import LayerStrategy

class LayerNormStrategy(LayerStrategy):
    """Strategy for INT8 Layer Normalization"""
    
    def __init__(self):
        pass
    
    def get_layer_type(self):
        return "layer_norm"
    
    def generate_config(self):
        # Input vector sizes (must match the feature dimension of your model)
        input_sizes = [128, 256, 512, 784, 1024]
        
        config = {
            'input_size': np.random.choice(input_sizes),
            'epsilon': 1, # Minimal value to prevent division by zero in integer space
            'data_range': (-128, 127),
            'weight_range': (-128, 127)
        }
        return config
    
    def generate_input_data(self, config):
        size = config['input_size']
        data_low, data_high = config['data_range']
        
        # Input features to normalize
        input_data = np.random.randint(data_low, data_high + 1, size=size, dtype=np.int8)
        
        # Gamma (scaling factor) and Beta (offset/shift)
        gamma = np.random.randint(data_low, data_high + 1, size=size, dtype=np.int8)
        beta = np.random.randint(data_low, data_high + 1, size=size, dtype=np.int8)
        
        return {
            'input': input_data,
            'gamma': gamma,
            'beta': beta
        }
    
    def compute_golden(self, data_dict, config):
        """
        Golden model for Layer Norm.
        Calculation: y = ((x - mean) / sqrt(var + eps)) * gamma + beta
        """
        # Cast to float for intermediate precision, then back to INT8
        # (Real hardware would use fixed-point/integer approximations)
        x = data_dict['input'].astype(np.float32)
        gamma = data_dict['gamma'].astype(np.float32)
        beta = data_dict['beta'].astype(np.float32)
        eps = config['epsilon']
        
        # 1. Calculate Mean
        mean = np.mean(x)
        
        # 2. Calculate Variance
        var = np.var(x)
        
        # 3. Normalize, Scale, and Shift
        # x_norm = (x - mean) / sqrt(var + eps)
        output = ((x - mean) / np.sqrt(var + eps)) * gamma + beta
        
        # --- INT8 SATURATION ---
        output_int8 = np.clip(np.round(output), -128, 127)
        
        return output_int8.astype(np.int8)