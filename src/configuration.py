import torch.nn as nn

from torch.optim.lr_scheduler import ReduceLROnPlateau

from .losses import CenterLoss

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
    def __init__(self, optimizer, patience:int=10, factor:float=0.1):
        self.patience = patience
        self.factor = factor
        self.scheduler = self.get_scheduler(optimizer)
        
    def get_scheduler(self, optimizer):
        return ReduceLROnPlateau(optimizer, mode='min', patience=self.patience, factor=self.factor)

    def get_last_lr(self):
        return self.scheduler.optimizer.param_groups[0]['lr']
        

class Configuration():
    def __init__(self, epochs:int, optimizer:str, early_stopping:EarlyStoppingConfig, reduce_lr:ReduceLROnPlateauConfig, criterion=nn.CrossEntropyLoss()):
        self.epochs = epochs
        self.optimizer = optimizer
        self.early_stopping = early_stopping
        self.reduce_lr = reduce_lr
        self.criterion = criterion

        self.has_early_stopping = self.early_stopping is not None
        self.has_reduce_lr = self.reduce_lr is not None

class AutoencoderConfiguration():
    def __init__(self, epochs:int, optimizer:str, early_stopping:EarlyStoppingConfig, reduce_lr:ReduceLROnPlateauConfig, alignment_loss=CenterLoss(), reconstruction_loss=nn.MSELoss(),align_weight:float=0.5, kl_weight:float=0.5, reconstruction_weight:float=0.0):
        self.epochs = epochs
        self.optimizer = optimizer
        self.early_stopping = early_stopping
        self.reduce_lr = reduce_lr
        self.alignment_loss = alignment_loss
        self.reconstruction_loss = reconstruction_loss
        
        self.align_weight = align_weight
        self.kl_weight = kl_weight
        self.reconstruction_weight = reconstruction_weight

        self.has_early_stopping = self.early_stopping is not None
        self.has_reduce_lr = self.reduce_lr is not None
    
    def calculate_loss(self, kl, x_recon, features, labels):
        rec_loss = self.reconstruction_loss(x_recon, features)
        align_loss = self.alignment_loss(x_recon, labels)
        return (rec_loss*self.reconstruction_weight) + (align_loss*self.align_weight) + (kl*self.kl_weight)