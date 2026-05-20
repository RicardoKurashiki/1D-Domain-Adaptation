import torch.nn as nn

class EarlyStoppingConfig():
    def __init__(self, patience:int=10, start_epoch:int=0, verbose:bool=True):
        self.patience = patience
        self.start_epoch = start_epoch
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False
        self.verbose = verbose

    def check(self, loss):
        if loss < self.best_loss:
            self.early_stop = False
            if self.verbose:
                print(f"Best val loss: {self.best_loss:.5f} -> {loss:.5f}")
            self.best_loss=loss
        else:
            self.counter += 1
            if self.verbose:
                print(f"No improvement. Patience {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                if self.verbose:
                    print("Patience limit reached. Stopping...")
                self.counter = 0
                self.early_stop=True

class ReduceLROnPlateauConfig():
    def __init__(self, patience:int=10, start_epoch:int=0, factor:float=0.1):
        pass

class Configuration():
    def __init__(self, epochs:int, optimizer:str, early_stopping:EarlyStoppingConfig, reduce_lr:ReduceLROnPlateauConfig, criterion=nn.CrossEntropyLoss()):
        self.epochs = epochs
        self.optimizer = optimizer
        self.early_stopping = early_stopping
        self.reduce_lr = reduce_lr
        self.criterion = criterion

        self.has_early_stopping = self.early_stopping is not None
        self.has_reduce_lr = self.reduce_lr is not None

