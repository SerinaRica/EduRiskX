from .transformer import TransformerPredictor

class SimpleTransformerPredictor(TransformerPredictor):
    def __init__(self, input_dim, max_len, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        # Initialize with baseline settings: No LayerNorm, No Class Weight, Fewer Layers/Heads/Dim
        super().__init__(
            input_dim=input_dim, 
            max_len=max_len, 
            d_model=d_model, 
            nhead=nhead, 
            num_layers=num_layers, 
            dropout=dropout,
            use_layernorm=False,
            use_class_weight=False
        )
