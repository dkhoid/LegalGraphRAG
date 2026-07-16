from datasets import load_dataset

datasets_to_check = ["adamwhite625/vietnam-legal-qa", "namphan1999/data-luat"]

for ds_name in datasets_to_check:
    try:
        print(f"\n{'='*50}")
        print(f"Loading dataset info for {ds_name}")

        # Try to load the train split or default
        dataset = load_dataset(ds_name, split="train", streaming=True)

        # Get a sample
        sample = next(iter(dataset))

        print(f"\nDataset Columns for {ds_name}:")
        print(list(sample.keys()))

        print("\nSample Data:")
        for k, v in sample.items():
            val = str(v)
            if len(val) > 200:
                val = val[:200] + "..."
            print(f"  {k}: {val}")

    except Exception as e:
        print(f"Error loading {ds_name}: {e}")
