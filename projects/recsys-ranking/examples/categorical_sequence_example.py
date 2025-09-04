"""
Example demonstrating categorical sequence feature support in FeatureEmbeddingDict.

This example shows how to use the FeatureEmbeddingDict with all three supported feature types:
- CATEGORICAL: Single categorical values (e.g., user_id, item_id)
- CATEGORICAL_SEQUENCE: Sequences of categorical values (e.g., user's item history)
- CONTINUOUS: Continuous numerical values (e.g., price, rating)
"""

import torch

from models.modules.feature_embedding_dict import FeatureEmbeddingDict
from my_types import FeatureSpec, FeatureType


def main():
    """Demonstrate categorical sequence feature support."""

    # Define feature specifications including categorical sequence
    feature_map = {
        # Single categorical feature (user ID)
        "user_id": FeatureSpec(
            type_=FeatureType.CATEGORICAL,
            embedding_dims=64,
            num_ids=10000,
            padding_idx=0,
        ),
        # Categorical sequence feature (user's viewing history)
        "item_history": FeatureSpec(
            type_=FeatureType.CATEGORICAL_SEQUENCE,
            embedding_dims=128,
            num_ids=50000,
            padding_idx=0,
            group_key="item_embedding",  # Share embeddings with other item features
        ),
        # Another categorical sequence feature sharing embeddings with item_history
        "item_candidates": FeatureSpec(
            type_=FeatureType.CATEGORICAL_SEQUENCE,
            embedding_dims=128,
            num_ids=50000,
            padding_idx=0,
            group_key="item_embedding",  # Share embeddings with item_history
        ),
        # Current target item (shares embeddings with sequences)
        "target_item": FeatureSpec(
            type_=FeatureType.CATEGORICAL,
            embedding_dims=128,
            num_ids=50000,
            padding_idx=0,
            group_key="item_embedding",  # Share embeddings with sequences
        ),
        # Continuous feature (price)
        "price": FeatureSpec(
            type_=FeatureType.CONTINUOUS,
            embedding_dims=32,
        ),
    }

    # Create the embedding dictionary
    embedding_dict = FeatureEmbeddingDict(feature_map)

    print("Created FeatureEmbeddingDict with mixed feature types:")
    print(f"Number of encoders: {len(embedding_dict.feature_encoder)}")
    print(f"Shared embeddings groups: {embedding_dict._group_key_dict}")

    # Create sample input data
    batch_size = 4
    seq_len = 10

    sample_inputs = {
        "user_id": torch.randint(1, 10000, (batch_size,)),
        "item_history": torch.randint(0, 50000, (batch_size, seq_len)),  # Some padding (0)
        "item_candidates": torch.randint(1, 50000, (batch_size, seq_len)),
        "target_item": torch.randint(1, 50000, (batch_size,)),
        "price": torch.randn(batch_size),
    }

    # Add some padding to demonstrate padding behavior
    sample_inputs["item_history"][0, -3:] = 0  # Last 3 positions are padding
    sample_inputs["item_history"][1, -1:] = 0  # Last position is padding

    print("\nSample input shapes:")
    for name, tensor in sample_inputs.items():
        print(f"  {name}: {tensor.shape}")

    # Forward pass
    with torch.no_grad():
        outputs = embedding_dict.forward(sample_inputs)

    print("\nOutput shapes:")
    for name, embedding in outputs.items():
        print(f"  {name}: {embedding.shape}")

    # Demonstrate shared embeddings
    print("\nShared embedding demonstration:")
    print(
        f"item_history encoder is target_item encoder: {embedding_dict.feature_encoder['item_history'] is embedding_dict.feature_encoder['target_item']}"
    )
    print(
        f"item_candidates encoder is target_item encoder: {embedding_dict.feature_encoder['item_candidates'] is embedding_dict.feature_encoder['target_item']}"
    )

    # Show that padding positions have zero embeddings
    print("\nPadding behavior demonstration:")
    history_embeddings = outputs["item_history"][0]  # First sample
    padding_positions = sample_inputs["item_history"][0] == 0

    if padding_positions.any():
        padding_embedding = history_embeddings[padding_positions][0]  # First padding position
        zero_norm = torch.norm(padding_embedding).item()
        print(f"Norm of embedding at padding position: {zero_norm:.6f} (should be ~0)")

    # Demonstrate different output shapes
    print("\nFeature type output differences:")
    print(f"  Categorical (user_id): {outputs['user_id'].shape} = (batch_size, embedding_dim)")
    print(
        f"  Categorical Sequence (item_history): {outputs['item_history'].shape} = (batch_size, seq_len, embedding_dim)"
    )
    print(f"  Continuous (price): {outputs['price'].shape} = (batch_size, embedding_dim)")

    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
