from abc import ABC, abstractmethod

class LayerStrategy(ABC):
    """Abstract base class for all layer types"""
    
    @abstractmethod
    def get_layer_type(self):
        """Return layer type: 'pool', 'conv', 'fc', etc."""
        pass

    @abstractmethod
    def generate_input_data(self, config):
        """Generate appropriate input data for this layer"""
        pass
    
    @abstractmethod
    def compute_golden(self, input_data, config):
        """Compute expected output using golden model"""
        pass
