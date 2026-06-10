import os
import numpy as np
import torch.nn as nn
import torch.optim as optim

from torchinfo import summary

from src.datasets import KermanyDataset, RSNADataset, FeatureSpaceDataset, MiniBatchSampler
from src.models import FeatureExtractor, ClassificationHead, ClassifierModel, Autoencoder
from src.configuration import Configuration, EarlyStoppingConfig
from src import utils, stage1, stage2, prototype

SEED = 42
BATCH_SIZE = 16
EPOCHS = 5
N_CLUSTERS = 2
LR = 0.0001

if __name__ == '__main__':
    utils.set_seed(SEED)

    train_dataset = KermanyDataset(split="train")
    val_dataset = KermanyDataset(split="val")
    test_dataset = KermanyDataset(split="test")

    print("Train: ", len(train_dataset))
    print("Val: ", len(val_dataset))
    print("Test: ", len(test_dataset))

    sampler = MiniBatchSampler(train_dataset, batch_size=BATCH_SIZE)

    extractor = FeatureExtractor(backbone="resnet18", unfrozen_layers=1)
    classifier = ClassificationHead(in_features=extractor.num_ftrs, out_features=2)
    model = ClassifierModel(extractor, classifier)

    summary(model, depth=10, col_names=["trainable"])

    print(f"\nExtractor size: {extractor.get_trainable_params()} params | {extractor.get_model_size():.3f} MB")
    print(f"Classifier size: {classifier.get_trainable_params()} params | {classifier.get_model_size():.3f} MB")
    print(f"Model size: {model.get_trainable_params()} params | {model.get_model_size():.3f} MB")

    train_data = utils.get_dataloader(train_dataset, sampler=sampler, shuffle=False)
    val_data = utils.get_dataloader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_data = utils.get_dataloader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    es = EarlyStoppingConfig()

    training_config = Configuration(epochs=EPOCHS, optimizer=optimizer, criterion=criterion, early_stopping=es, reduce_lr=None)

    stage1.train(path="./", model=model, train_data=train_data, val_data=val_data, config=training_config)
    stage1.test(path="./", model=model, data=test_data, criterion=criterion)
    features, labels = stage1.extract_features(path="./", model=extractor, data=train_data)

    features = np.load(os.path.join("./", f"features.npy"))
    labels = np.load(os.path.join("./", f"labels.npy"))

    feature_dataset = FeatureSpaceDataset(features=features, labels=labels)
    feature_data = utils.get_dataloader(feature_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    stage2.test_features(path="./", model=classifier, data=feature_data, criterion=criterion)

    # pt_dataset = prototype.run(dataset=feature_dataset, undersampling="enn", clustering="k_means", k=N_CLUSTERS, seed=SEED)