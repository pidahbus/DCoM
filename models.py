from transformers import TFDistilBertForSequenceClassification, DistilBertConfig
from tensorflow.keras import layers as L
from tensorflow.keras.models import Model
import tensorflow as tf
import pandas as pd
from transformers import DistilBertTokenizer
from generator import BatchGenerator
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras import callbacks as C
from callbacks import AverageF1
import os
import pickle
from loguru import logger
from utils import create_aux_features, clean_data, normalize_data, calculate_mean_and_std_of_cont_vars
import numpy as np
from tqdm import tqdm
from collections import Counter


class DCoMModel:
    def __init__(self, use_aux, model_type, padding_len, vocab_size):
        self.use_aux = use_aux
        self.model_type = model_type
        self.padding_len = padding_len
        self.vocab_size = vocab_size

    def __nPrDistilBert__(self):
        if self.use_aux:
            bertmodel = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased',
                                                                              config=DistilBertConfig.from_pretrained(
                                                                                  'distilbert-base-uncased',
                                                                                  num_labels=256))
            id_input = L.Input(shape=(self.padding_len,), name="input_ids", dtype=tf.int32)
            attn_input = L.Input(shape=(self.padding_len,), name="attention_mask", dtype=tf.int32)
            aux_input = L.Input(shape=(len(self.aux_features), ), name="aux_input")

            bertout = bertmodel({"input_ids": id_input, "attention_mask": attn_input}).logits
            aux_x = L.Dense(64, activation="relu")(aux_input)
            aux_x = L.Dropout(rate=0.3)(aux_x)

            x = L.Concatenate()([bertout, aux_x])
            x = L.Dense(256, activation="relu")(x)
            x = L.Dropout(rate=0.3)(x)
            x = L.BatchNormalization()(x)

            y = L.Dense(len(self.label_to_idx), activation="softmax")(x)

            model = Model({"input_ids": id_input,
                           "attention_mask": attn_input,
                           "aux": aux_input}, y)
            print(model.summary())
        else:
            model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased',
                                                                          config=DistilBertConfig.from_pretrained(
                                                                              'distilbert-base-uncased',
                                                                              num_labels=len(self.label_to_idx)))
        return model

    def __nPrLSTM__(self, vocab_size):
        col_input = tf.keras.layers.Input(shape=(None,))
        x = tf.keras.layers.Embedding(vocab_size, 768, mask_zero=True)(col_input)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512))(x)
        x = tf.keras.layers.Dense(512, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.3)(x)

        if self.use_aux:
            aux_input = tf.keras.layers.Input(shape=(len(self.aux_features),))
            stat_x = tf.keras.layers.Dense(128, activation="relu")(aux_input)
            stat_x = tf.keras.layers.Dropout(0.3)(stat_x)
            x = tf.keras.layers.Concatenate()([x, stat_x])

        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.BatchNormalization()(x)

        y = tf.keras.layers.Dense(len(self.label_to_idx), activation="softmax")(x)

        if self.use_aux:
            model = Model({"input_ids": col_input, "aux": aux_input}, y)
        else:
            model = Model({"input_ids": col_input}, y)

        print(model.summary())
        return model

    def load_model(self, checkpoint):
        self.epoch = 0
        if self.model_type == "nPr-DistilBert":
            self.model = self.__nPrDistilBert__()
        elif self.model_type == "nPr-LSTM":
            self.model = self.__nPrLSTM__(vocab_size=self.vocab_size)

        if checkpoint:
            self.model.load_weights(checkpoint)
            try:
                self.epoch = int(checkpoint.split("/")[-1].replace(".h5", ""))
            except:
                pass

    def load_metadata(self, metadata_path):
        metadata = pd.read_pickle(metadata_path)
        self.tokenizer = metadata["tokenizer"]
        self.label_to_idx = metadata["label_to_idx"]
        self.idx_to_label = metadata["idx_to_label"]
        self.aux_features = None
        if self.use_aux:
            self.aux_features = metadata["aux_features"]
            self.mean_cont_vars = metadata["mean_cont_vars"]
            self.std_cont_vars = metadata["std_cont_vars"]

    def fit(self, train, valid, batch_size, optimizer, model_save_dir, init_checkpoint_path, init_metadata_path):

        logger.info("Cleaning Data")
        train = clean_data(train)
        valid = clean_data(valid)

        if init_metadata_path:
            logger.info("Loading Metadata")
            self.load_metadata(init_metadata_path)
        else:
            logger.info("Creating Metadata")
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            self.label_to_idx = dict(zip(train.type.unique().tolist(), list(range(train.type.nunique()))))
            self.idx_to_label = {}
            self.aux_features = None
            for l, idx in self.label_to_idx.items():
                self.idx_to_label[idx] = l

        if self.use_aux:
            train, self.aux_features = create_aux_features(train)
            valid, _ = create_aux_features(valid)
            if not init_metadata_path:
                self.mean_cont_vars, self.std_cont_vars = calculate_mean_and_std_of_cont_vars(train, self.aux_features)
            train = normalize_data(train, self.mean_cont_vars, self.std_cont_vars, self.aux_features)
            valid = normalize_data(valid, self.mean_cont_vars, self.std_cont_vars, self.aux_features)

        logger.info("Loading Model")
        self.load_model(init_checkpoint_path)


        logger.info("Creating Batch Generator")
        train_gen = BatchGenerator(train, batch_size, self.padding_len, "train", self.aux_features, self.label_to_idx,
                                   self.tokenizer, self.model_type, self.use_aux)
        loss = SparseCategoricalCrossentropy(from_logits=True)
        metric = SparseCategoricalAccuracy("accuracy")

        logger.info("Compiling Model")
        self.model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

        checkpoint = C.ModelCheckpoint(os.path.join(model_save_dir, "{epoch}.h5"), save_weights_only=True)
        logs = C.CSVLogger(os.path.join(model_save_dir, "logs.csv"))
        reduce_lr = C.ReduceLROnPlateau(factor=0.5, monitor="test_avg_f1", patience=2)
        avg_f1 = AverageF1(valid, 10, batch_size, self.padding_len, "valid", self.aux_features, self.label_to_idx,
                           self.tokenizer, self.model_type, self.use_aux)

        metadata = {
            "tokenizer": self.tokenizer,
            "idx_to_label": self.idx_to_label,
            "label_to_idx": self.label_to_idx
        }
        if self.use_aux:
            metadata.update({
                "aux_features": self.aux_features,
                "mean_cont_vars": self.mean_cont_vars,
                "std_cont_vars": self.std_cont_vars
            })

        logger.info("Saving Metadata")
        with open(os.path.join(model_save_dir, "metadata.pkl"), "wb") as m:
            pickle.dump(metadata, m)

        logger.info("Training Started")
        self.model.fit(train_gen, epochs=1000, callbacks=[avg_f1, checkpoint, logs, reduce_lr],
                       initial_epoch=self.epoch)

    def predict(self, test, batch_size, samples_per_instance):
        test = clean_data(test)
        if self.use_aux:
            test, _ = create_aux_features(test)
            test = normalize_data(test, self.mean_cont_vars, self.std_cont_vars, self.aux_features)

        test_pred_arr = np.zeros((test.shape[0], samples_per_instance))
        for t in tqdm(range(samples_per_instance)):
            test_gen = BatchGenerator(test, batch_size, self.padding_len, "test", self.aux_features, self.label_to_idx,
                                      self.tokenizer, self.model_type, self.use_aux)
            preds = self.model.predict(test_gen)
            pred_class = np.argmax(preds, axis=1)
            test_pred_arr[:, t] = pred_class

        test_pred_class = []
        for i in range(test_pred_arr.shape[0]):
            test_pred_class.append(Counter(test_pred_arr[i,]).most_common()[0][0])

        # test_out = test.copy()
        # test_out["pred-type"] = [self.idx_to_label[idx] for idx in test_pred_class]
        # return test_out
        return [self.idx_to_label[idx] for idx in test_pred_class]