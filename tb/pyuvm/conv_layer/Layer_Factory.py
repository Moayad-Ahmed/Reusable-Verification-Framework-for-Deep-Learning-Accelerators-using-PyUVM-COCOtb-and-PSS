from convolution_strategy import ConvolutionStrategy

class LayerFactory:
    _strategies = {
        'convolution': ConvolutionStrategy
        # Add more as you implement them
    }
    
    @classmethod
    def create_strategy(cls, layer_type):
        """Create a strategy instance"""
        if layer_type not in cls._strategies:
            raise ValueError(f"Unknown layer type: {layer_type}")
        
        return cls._strategies[layer_type]()
    
    @classmethod
    def register_strategy(cls, layer_type, strategy_class):
        """Register a new layer strategy"""
        cls._strategies[layer_type] = strategy_class
    
    @classmethod
    def get_available_layers(cls):
        """List all available layer types"""
        return list(cls._strategies.keys())
