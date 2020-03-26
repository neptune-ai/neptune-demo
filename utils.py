import numpy as np


def log_data(logs, neptune):
    neptune.log_metric('epoch_accuracy', logs['accuracy'])
    neptune.log_metric('epoch_loss', logs['loss'])


def lr_scheduler(epoch, neptune, learning_rate):
    if epoch < 20:
        new_lr = learning_rate
    else:
        new_lr = learning_rate * np.exp(0.05 * (20 - epoch))

    neptune.log_metric('learning_rate', new_lr)
    return new_lr
