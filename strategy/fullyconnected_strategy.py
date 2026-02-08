import numpy as np
from LayerStrategy import LayerStrategy

class FullyConnectedStrategy(LayerStrategy):
    """Strategy for fully connected (linear/dense) layers with separated data generation"""
    
    def __init__(self):
        pass
    
    def get_layer_type(self):
        return "fully_connected"

    def generate_input_weights(self, config):
        """Generates the weight matrix [output_size, input_size]"""
        weight_low, weight_high = config['weight_range']
        shape = (config['output_size'], config['input_size'])
        weights = np.random.randint(weight_low, weight_high + 1, size=shape)
        return weights.astype(np.int8)

    def generate_input_biases(self, config):
        """Generates the bias vector [output_size]"""
        if config['use_bias']:
            data_low, data_high = config['data_range']
            shape = (config['output_size'],)
            bias = np.random.randint(data_low, data_high + 1, size=shape)
            return bias.astype(np.int8)
        else:
            return np.zeros(config['output_size'], dtype=np.int8)

    def generate_input_data(self, config):
        """Generates only the input feature vector [input_size]"""
        data_low, data_high = config['data_range']
        shape = (config['input_size'],)
        input_data = np.random.randint(data_low, data_high + 1, size=shape)
        return input_data.astype(np.int8)
    
    def compute_golden(self, input_data, weights, bias, config):
        """
        Calculates reference output using three separate input variables.
        
        Args:
            input_data: 1D array of features
            weights: 2D array of weights
            bias: 1D array of biases
            config: Layer configuration dictionary
        """
        # Compute full-precision dot product, then truncate to int8 (wrapping, no saturation)
        output = np.matmul(weights.astype(np.int32), input_data.astype(np.int32))
        
        if config['use_bias']:
            output = output + bias.astype(np.int32)
        
        # Truncate to int8 with wrapping (matches 8-bit DUT accumulator)
        return output.astype(np.int64)