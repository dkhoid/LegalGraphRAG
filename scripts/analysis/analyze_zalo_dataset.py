import json
from datasets import load_dataset_builder, load_dataset

try:
    print("Loading dataset info for GreenNode/zalo-ai-legal-text-retrieval-vn (corpus)")

    # Load dataset corpus
    dataset = load_dataset(
        "GreenNode/zalo-ai-legal-text-retrieval-vn", "corpus", split="test", streaming=True
    )

    # Get a sample
    sample = next(iter(dataset))

    print("\nDataset Columns:")
    print(list(sample.keys()))

    print("\nSample Data:")
    for k, v in sample.items():
        val = str(v)
        if len(val) > 200:
            val = val[:200] + "..."
        print(f"  {k}: {val}")

except Exception as e:
    print(f"Error: {e}")
