from transformers import (
    LongformerTokenizer,
    LongformerConfig,
    LongformerModel,
    LongformerForMaskedLM,
)

config = LongformerConfig(max_position_embeddings=4098)
