PADDING_LEN = 300
BATCH_SIZE = 32
USE_AUX = True
MODEL_TYPE =  "nPr-LSTM" #"nPr-DistilBert" #nPr-LSTM
VOCAB_SIZE = 30522
OPTIMIZER = "adam"


TRAIN_CSV_PATH = "./data/train.csv"
VALID_CSV_PATH = "./data/valid.csv"
TEST_CSV_PATH = "./data/test.csv"

MODEL_SAVE_DIR = "./checkpoint"
INITIAL_MODEL_WEIGHT_PATH = "./checkpoint/1.h5"
INITIAL_METADATA_PATH = "./checkpoint/metadata.pkl"

TEST_SAMPLES_PER_INSTANCE = 2



