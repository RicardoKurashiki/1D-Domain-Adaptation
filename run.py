from src.datasets import KermanyDataset, MiniBatchSampler
from src import utils
import pandas as pd

utils.set_seed(42)

train_dataset = KermanyDataset(split="train")
val_dataset = KermanyDataset(split="val")
test_dataset = KermanyDataset(split="test")

print("Train: ", len(train_dataset))
print("Val: ", len(val_dataset))
print("Test: ", len(test_dataset))

sampler = MiniBatchSampler(train_dataset, batch_size=4)

for i in range(10):
    print("iter: ", i)
    results = []
    for batch in sampler:
        result = {"batch": batch, "idx": []}
        for index in batch:
            c = train_dataset.data.iloc[index]
            result["idx"].append({"idx": index, "label": c["label"], "path": c["path"]})
        results.append(result)

    df = pd.DataFrame(data=results)
    df.to_csv(f"batches_{i+1}.csv", index=False)