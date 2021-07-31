from tensorflow.keras.utils import Sequence
from random import sample, choices
import numpy as np


class BatchGenerator(Sequence):
    def __init__(self, data, batch_size, padding_len, mode, aux_features, label_to_idx, tokenizer, model_type, use_aux):
        self.data = data
        self.batch_size = batch_size
        self.padding_len = padding_len
        self.label_to_idx = label_to_idx
        self.mode = mode
        self.aux_features = aux_features
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.use_aux = use_aux

    def __len__(self):
        if len(self.data) % self.batch_size == 0:
            return len(self.data) // self.batch_size
        return len(self.data) // self.batch_size + 1

    def __getitem__(self, idx):
        batch_slice = slice(idx * self.batch_size, (idx + 1) * self.batch_size)
        data_batch = self.data[batch_slice].copy(deep=True).reset_index(drop=True)
        if self.mode == "train":
            data_batch["values"] = data_batch["values"].apply(lambda x: list(self.tokenizer.encode_plus(
                " [SEP] ".join(choices(x, k=sample(list(range(1, np.min([200, len(x)]) + 1)), k=1)[0])),
                add_special_tokens=True,
                max_length=self.padding_len,
                padding="max_length",
                truncation=True,
                return_attention_mask=True).values()))
        else:
            data_batch["values"] = data_batch["values"].apply(
                lambda x: list(self.tokenizer.encode_plus(" [SEP] ".join(choices(x, k=min(len(x), 200))),
                                                          add_special_tokens=True,
                                                          max_length=self.padding_len,
                                                          padding="max_length",
                                                          truncation=True,
                                                          return_attention_mask=True).values()))
        data_batch["type"] = data_batch["type"].apply(lambda x: self.label_to_idx[x])

        zipped_content = list(zip(*data_batch["values"].tolist()))
        input_ids_list = list(zipped_content[0])

        x_out = {"input_ids": np.array(input_ids_list)}

        if self.model_type == "nPr-DistilBert":
            attention_mask_list = list(zipped_content[1])
            x_out["attention_mask"] = np.array(attention_mask_list)

        if self.use_aux:
            aux = np.array(data_batch[self.aux_features])
            x_out["aux"] = aux

        label_list = np.array(data_batch["type"]).reshape(data_batch.shape[0], 1).tolist()
        return x_out, np.array(label_list)


    def on_epoch_end(self):
        self.data = self.data.sample(frac=1).reset_index(drop=True)
