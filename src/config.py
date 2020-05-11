import transformers


MAX_LEN = 50
TRAIN_BATCH_SIZE = 6
VALID_BATCH_SIZE = 3
EPOCHS = 5
BERT_PATH = "../inputs/bert_base_uncased/"
MODEL_PATH = "model.bin"
TRAINING_FILE = "../inputs/data.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BERT_PATH,
    do_lower_case=True
)
