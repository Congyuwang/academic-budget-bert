OUT_PATH=../vocab
mkdir -p $OUT_PATH
curl --retry 9999 -C - -o ../vocab/bert-large-uncased-vocab.txt https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt
