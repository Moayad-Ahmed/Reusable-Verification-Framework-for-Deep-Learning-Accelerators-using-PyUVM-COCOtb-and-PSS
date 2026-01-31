from LayerStrategy import LayerStrategy
import torch
import torch.nn as nn
from cocotb.triggers import RisingEdge
import numpy as np
import random

class FullyConnectedStrategy(LayerStrategy):
    """Strategy for fully connected (linear/dense) layers"""
    
    def __init__(self):
        pass
    
    def get_layer_type(self):
        return "fully_connected"
    
    def generate_config(self):
        """
        Generate random but valid FC layer configuration.
        """
        # Common FC layer sizes in neural networks
        input_sizes = [128, 256, 512, 784, 1024]
        output_sizes = [10, 64, 128, 256, 512]
        
        config = {
            'input_size': np.random.choice(input_sizes),
            'output_size': np.random.choice(output_sizes),
            'use_bias': True,
            'data_range': (-128, 127),  # 8-bit signed data
            'weight_range': (-128, 127)  # 8-bit signed weights
        }
        return config
    
    def generate_input_data(self, config):
        """
        Generate random input vector, weights, and bias for FC layer.
        
        config = {
            'input_size': 256,
            'output_size': 10,
            'use_bias': True,
            'data_range': (-128, 127),
            'weight_range': (-128, 127)
        }
        
        Returns dict with:
            - 'input': input vector [input_size]
            - 'weights': weight matrix [output_size, input_size]
            - 'bias': bias vector [output_size]
        """
        input_size = config['input_size']
        output_size = config['output_size']
        data_low, data_high = config['data_range']
        weight_low, weight_high = config['weight_range']
        
        # Generate random input vector (1D)
        input_data = np.random.randint(data_low, data_high + 1, size=input_size)
        
        # Generate random weights (2D matrix: output_size x input_size)
        weights = np.random.randint(weight_low, weight_high + 1, 
                                   size=(output_size, input_size))
        
        # Generate random bias (1D)
        if config['use_bias']:
            bias = np.random.randint(data_low, data_high + 1, size=output_size)
        else:
            bias = np.zeros(output_size, dtype=np.int32)
        
        return {
            'input': input_data.astype(np.int32),
            'weights': weights.astype(np.int32),
            'bias': bias.astype(np.int32)
        }
    
    def compute_golden(self, data_dict, config):
        """
        Golden model that matches 8-bit Integer Hardware for FC layer.
        
        Computation: output = (input @ weights.T) + bias
        
        Uses pure integer arithmetic to match hardware exactly.
        """
        input_data = data_dict['input']
        weights = data_dict['weights']
        bias = data_dict['bias']
        
        # Matrix multiply: output[i] = sum(input[j] * weights[i][j]) for all j
        # Shape: (output_size,) = (output_size, input_size) @ (input_size,)
        output = np.matmul(weights, input_data)  # Integer multiplication
        
        # Add bias
        if config['use_bias']:
            output = output + bias
        
        # Clip to valid 32-bit accumulator range
        output = np.clip(output, -(2**31), 2**31 - 1)
        
        return output.astype(np.int32)