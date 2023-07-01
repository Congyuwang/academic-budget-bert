# !pip install tokenizers
from tokenizers import BertWordPieceTokenizer
import os

# initialize
tokenizer = BertWordPieceTokenizer(
    clean_text=True,
    handle_chinese_chars=False,
    strip_accents=True,
    lowercase=True
)
paths = "./exported_text.txt"
# and train
tokenizer.train(files=paths, vocab_size=30000, min_frequency=2,
                limit_alphabet=1000, wordpieces_prefix='##',
                special_tokens=[
                    '[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'])

tokenizer.save("./new_vocab/vocab.json")