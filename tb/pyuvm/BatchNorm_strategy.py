import numpy as np
from LayerStrategy import LayerStrategy

class BatchNormStrategy(LayerStrategy):
    """Strategy for INT8 Batch Normalization (Inference Mode)"""
    
    def __init__(self):
        pass
    
    def get_layer_type(self):
        return "batch_norm"
    
    def generate_config(self):
        """
        Generate configuration for BatchNorm.
        In BatchNorm, the input_size refers to the number of channels/features.
        """
        input_sizes = [16, 32, 64, 128, 256, 512]
        
        config = {
            'input_size': np.random.choice(input_sizes),
            'epsilon': 1e-5,  # Small constant to avoid division by zero
            'data_range': (-128, 127),
            'weight_range': (-128, 127)
        }
        return config
    
    def generate_input_data(self, config):
        """
        Generates the current input vector and the pre-calculated 
        parameters (mean, var, gamma, beta) from training.
        """
        size = config['input_size']
        data_low, data_high = config['data_range']
        
        # Current input features to be normalized
        input_data = np.random.randint(data_low, data_high + 1, size=size, dtype=np.int8)
        
        # Pre-calculated Running Mean and Variance (learned during training)
        running_mean = np.random.randint(-64, 64, size=size, dtype=np.int8)
        # Variance must be positive; using a range that avoids zero
        running_var = np.random.randint(1, 127, size=size, dtype=np.int8)
        
        # Learnable scaling (gamma) and shifting (beta) parameters
        gamma = np.random.randint(data_low, data_high + 1, size=size, dtype=np.int8)
        beta = np.random.randint(data_low, data_high + 1, size=size, dtype=np.int8)
        
        return {
            'input': input_data,
            'mean': running_mean,
            'var': running_var,
            'gamma': gamma,
            'beta': beta
        }
    
    def compute_golden(self, data_dict, config):
        """
        Golden model for BatchNorm inference.
        Formula: y = ((x - mean) / sqrt(var + eps)) * gamma + beta
        """
        # Convert to float for intermediate high-precision math
        x = data_dict['input'].astype(np.float32)
        mu = data_dict['mean'].astype(np.float32)
        var = data_dict['var'].astype(np.float32)
        gamma = data_dict['gamma'].astype(np.float32)
        beta = data_dict['beta'].astype(np.float32)
        eps = config['epsilon']
        
        # 1. Normalize
        normalized = (x - mu) / np.sqrt(var + eps)
        
        # 2. Scale and Shift (Affine transform)
        output = (normalized * gamma) + beta
        
        # 3. Round to nearest integer and clip to signed INT8 range [-128, 127]
        output_int8 = np.clip(np.round(output), -128, 127)
        
        return output_int8.astype(np.int8)