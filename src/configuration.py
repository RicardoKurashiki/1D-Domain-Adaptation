class Configuration():
    def __init__(self, epochs:int, batch_size:int, optimizer:str, learning_rate:float, early_stopping:SchedulerConfig, reduce_lr:SchedulerConfig):
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.early_stopping = early_stopping
        self.reduce_lr = reduce_lr

        self.has_early_stopping = self.early_stopping is not None
        self.has_reduce_lr = self.reduce_lr is not None

class EarlyStoppingConfig():
    def __init__(self, patience:int=10, start_epoch:int=0, verbose:bool=True):
        self.mode = mode
        self.patience = patience
        self.start_epoch = start_epoch
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False

    def check(self, loss):
        if loss < self.best_loss:
            self.early_stop = False
            if verbose:
                print(f"Best val loss: {self.best_loss:.5f} -> {loss:.5f}")
            self.best_loss=loss
        else:
            self.counter += 1
            if verbose:
                print(f"No improvement. Patience {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                if verbose:
                    print("Patience limit reached. Stopping...")
                self.counter = 0
                self.early_stop=True

class ReduceLROnPlateauConfig():
    def __init__(self, patience:int=10, start_epoch:int=0, factor:float=0.1):
        pass
