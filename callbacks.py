from tensorflow.keras.callbacks import Callback
from generator import BatchGenerator
import numpy as np
from sklearn.metrics import f1_score


class AverageF1(Callback):
    def __init__(self, data, patience, batch_size, padding_len, mode, aux_features, label_dict, tokenizer, model_type,
                 use_aux):
        super(AverageF1, self).__init__()
        self.data = data
        self.patience = patience
        self.best = 0
        self.wait = 0
        self.batch_size = batch_size
        self.padding_len = padding_len
        self.mode = mode
        self.aux_features = aux_features
        self.label_dict = label_dict
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.use_aux = use_aux

    def on_epoch_end(self, epoch, logs=None):
        test_gen = BatchGenerator(self.data, self.batch_size, self.padding_len, self.mode, self.aux_features,
                                  self.label_dict, self.tokenizer, self.model_type, self.use_aux)

        preds = self.model.predict(test_gen)
        if type(preds) == tuple:
            preds = preds[0]
        pred_class = np.argmax(preds, axis=1).reshape(-1)
        true_class = self.data["type"].apply(lambda x: self.label_dict[x])
        average_f1 = f1_score(true_class, pred_class, average="weighted")

        print(f" - test_avg_f1: {average_f1}")
        logs["test_avg_f1"] = average_f1

        if np.greater(average_f1, self.best):
            self.best = average_f1
            self.wait = 0

        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
