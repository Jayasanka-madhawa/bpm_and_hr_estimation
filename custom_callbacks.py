from keras.callbacks import Callback

class EarlyStoppingIncreasingValLoss(Callback):
    def __init__(self, patience=0, delta=0, check_interval=10, increase_threshold=0):
        super(EarlyStoppingIncreasingValLoss, self).__init__()
        self.patience = patience
        self.delta = delta
        self.check_interval = check_interval
        self.increase_threshold = increase_threshold
        self.best_val_loss = float('inf')
        self.wait = 0
        self.num_checks = 0

    def on_epoch_end(self, epoch, logs=None):
        current_val_loss = logs.get('val_loss')
        if current_val_loss is None:
            return

        self.num_checks += 1
        if self.num_checks % self.check_interval == 0:
            loss_increase = current_val_loss - self.best_val_loss
            if loss_increase > self.increase_threshold:
                self.wait += 1
                if self.wait >= self.patience:
                    self.model.stop_training = True
                    print(f'\nEarly stopping due to validation loss increase.')
            else:
                self.wait = 0
            self.best_val_loss = current_val_loss
            
# class EarlyStoppingIncreasingValLoss(Callback):
#     def __init__(self, patience=0, delta=0):
#         super(EarlyStoppingIncreasingValLoss, self).__init__()
#         self.patience = patience
#         self.delta = delta
#         self.best_val_loss = float('inf')
#         self.wait = 0

#     def on_epoch_end(self, epoch, logs=None):
#         current_val_loss = logs.get('val_loss')
#         if current_val_loss is None:
#             return

#         if current_val_loss > self.best_val_loss + self.delta:
#             self.wait += 1
#             print(f'count self.wait :{self.wait}')
#             if self.wait >= self.patience:
#                 self.model.stop_training = True
#                 print(f'\nEarly stopping due to increasing validation loss.')
#         else:
#             self.best_val_loss = current_val_loss
#             self.wait = 0

            # early_stopping = EarlyStoppingIncreasingValLoss(patience=10, delta=0.01)    
