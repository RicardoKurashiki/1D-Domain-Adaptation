from torchinfo import summary

from src.datasets import KermanyDataset, MiniBatchSampler
from src.models import FeatureExtractor, ClassificationHead, ClassifierModel
from src import utils

utils.set_seed(42)

train_dataset = KermanyDataset(split="train")
val_dataset = KermanyDataset(split="val")
test_dataset = KermanyDataset(split="test")

print("Train: ", len(train_dataset))
print("Val: ", len(val_dataset))
print("Test: ", len(test_dataset))

sampler = MiniBatchSampler(train_dataset, batch_size=4)

extractor = FeatureExtractor(backbone="resnet18", unfrozen_layers=3)
classifier = ClassificationHead(in_features=extractor.num_ftrs, out_features=2)
model = ClassifierModel(extractor, classifier)

summary(extractor, depth=10, col_names=["trainable"])