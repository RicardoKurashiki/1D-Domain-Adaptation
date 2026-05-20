from torchinfo import summary

from src.datasets import RSNADataset, MiniBatchSampler
from src.models import FeatureExtractor, ClassificationHead, ClassifierModel
from src import utils

utils.set_seed(42)

train_dataset = RSNADataset(split="train")
val_dataset = RSNADataset(split="val")
test_dataset = RSNADataset(split="test")

print("Train: ", len(train_dataset))
print("Val: ", len(val_dataset))
print("Test: ", len(test_dataset))

sampler = MiniBatchSampler(train_dataset, batch_size=4)

extractor = FeatureExtractor(backbone="resnet18", unfrozen_layers=3)
classifier = ClassificationHead(in_features=extractor.num_ftrs, out_features=2)
model = ClassifierModel(extractor, classifier)

summary(model, depth=10, col_names=["trainable"])

print(f"\nExtractor size: {extractor.get_trainable_params()} params | {extractor.get_model_size():.3f} MB")
print(f"Classifier size: {classifier.get_trainable_params()} params | {classifier.get_model_size():.3f} MB")
print(f"Model size: {model.get_trainable_params()} params | {model.get_model_size():.3f} MB")