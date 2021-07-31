import numpy as np
from collections import Counter
import string
from scipy.stats import entropy, skew, kurtosis
from tqdm import tqdm
tqdm.pandas()


def create_aux_features(df):
    df["num_values"] = df["values"].progress_apply(len)
    df["col_entropy"] = df["values"].progress_apply(lambda x: entropy(np.array(list(Counter(x).values())) / len(x)))

    df["numeric_char_fraction"] = df["values"].progress_apply(
        lambda x: np.mean([any([c.isdigit() for c in i]) for i in x]))
    df["numeric_char_mean"] = df["values"].progress_apply(
        lambda x: np.mean([np.sum([c.isdigit() for c in i]) for i in x]))
    df["numeric_char_std"] = df["values"].progress_apply(
        lambda x: np.std([np.sum([c.isdigit() for c in i]) for i in x]))

    df["alpha_char_fraction"] = df["values"].progress_apply(
        lambda x: np.mean([any([c.isalpha() for c in i]) for i in x]))
    df["alpha_char_mean"] = df["values"].progress_apply(
        lambda x: np.mean([np.sum([c.isalpha() for c in i]) for i in x]))
    df["alpha_char_std"] = df["values"].progress_apply(lambda x: np.std([np.sum([c.isalpha() for c in i]) for i in x]))

    df["special_char_mean"] = df["values"].progress_apply(
        lambda x: np.mean([sum([v for k, v in Counter(i).items() if k in string.punctuation]) for i in x]))
    df["special_char_std"] = df["values"].progress_apply(
        lambda x: np.std([sum([v for k, v in Counter(i).items() if k in string.punctuation]) for i in x]))

    df["word_count_mean"] = df["values"].progress_apply(lambda x: np.mean([len(i.split()) for i in x]))
    df["word_count_std"] = df["values"].progress_apply(lambda x: np.std([len(i.split()) for i in x]))

    df["value_len_sum"] = df["values"].progress_apply(lambda x: np.sum([len(i) for i in x]))
    df["value_len_min"] = df["values"].progress_apply(lambda x: np.min([len(i) for i in x]))
    df["value_len_max"] = df["values"].progress_apply(lambda x: np.max([len(i) for i in x]))
    df["value_len_median"] = df["values"].progress_apply(lambda x: np.median([len(i) for i in x]))
    df["value_len_mode"] = df["values"].progress_apply(lambda x: Counter([len(i) for i in x]).most_common(1)[0][0])
    df["value_len_skewness"] = df["values"].progress_apply(lambda x: skew([len(i) for i in x]))
    df["value_len_kurtosis"] = df["values"].progress_apply(lambda x: kurtosis([len(i) for i in x]))

    aux_features = df.columns.tolist()[2:]
    assert len(aux_features) == 19

    return df, aux_features


def calculate_mean_and_std_of_cont_vars(df, cont_vars):
    mean_cont_vars = df[cont_vars].mean()
    std_cont_vars = df[cont_vars].std()
    return mean_cont_vars, std_cont_vars


def clean_data(df):
    df = df[df["values"].apply(lambda x: len(x) > 0)].reset_index(drop=True)
    df["values"] = df["values"].apply(lambda x: list(map(str, x)))
    df["values"] = df["values"].apply(lambda x: list(set(x)))
    return df


def normalize_data(df, mean_cont_vars=None, std_cont_vars=None, cont_vars=None):
    df[cont_vars] -= mean_cont_vars
    df[cont_vars] /= std_cont_vars
    return df
