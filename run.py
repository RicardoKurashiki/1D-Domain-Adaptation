import torch.nn as nn
import torch.optim as optim

from torchinfo import summary

from src.datasets import KermanyDataset, RSNADataset, MiniBatchSampler
from src.models import FeatureExtractor, ClassificationHead, ClassifierModel, Autoencoder
from src.configuration import Configuration, EarlyStoppingConfig
from src import utils, stage1


if __name__ == '__main__':
    utils.set_seed(42)

    train_dataset = KermanyDataset(split="train")
    val_dataset = KermanyDataset(split="val")
    test_dataset = KermanyDataset(split="test")

    print("Train: ", len(train_dataset))
    print("Val: ", len(val_dataset))
    print("Test: ", len(test_dataset))

    sampler = MiniBatchSampler(train_dataset, batch_size=32)

    extractor = FeatureExtractor(backbone="resnet18", unfrozen_layers=1)
    classifier = ClassificationHead(in_features=extractor.num_ftrs, out_features=2)
    model = ClassifierModel(extractor, classifier)

    summary(model, depth=10, col_names=["trainable"])

    print(f"\nExtractor size: {extractor.get_trainable_params()} params | {extractor.get_model_size():.3f} MB")
    print(f"Classifier size: {classifier.get_trainable_params()} params | {classifier.get_model_size():.3f} MB")
    print(f"Model size: {model.get_trainable_params()} params | {model.get_model_size():.3f} MB")

    train_data = utils.get_dataloader(train_dataset, sampler=sampler, shuffle=False)
    val_data = utils.get_dataloader(val_dataset, batch_size=32, shuffle=False)
    test_data = utils.get_dataloader(test_dataset, batch_size=32, shuffle=False)

    lr = 0.0001

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    es = EarlyStoppingConfig()

    training_config = Configuration(epochs=5, optimizer=optimizer, criterion=criterion, early_stopping=es, reduce_lr=None)

    # stage1.train(path="./", model=model, train_data=train_data, val_data=val_data, config=training_config)
    stage1.test(path="./", model=model, data=test_data, criterion=criterion)

    target_data = utils.get_dataloader(RSNADataset(split="test"), batch_size=32, shuffle=False)
    stage1.test(path="./", model=model, data=target_data, criterion=criterion)

    autoencoder = Autoencoder("simple_autoencoder", model.extractor.num_ftrs, model.extractor.num_ftrs//2, model.extractor.num_ftrs//4, 2)
    summary(autoencoder, depth=10, col_names=["trainable"])