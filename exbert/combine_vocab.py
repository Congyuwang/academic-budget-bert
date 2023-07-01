import json

def extract_vocab(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        vocab = data['model']['vocab']
        return list(vocab.keys())

def merge_vocabularies(vocab_file1, vocab_file2, merged_vocab_file):
    vocab1 = extract_vocab(vocab_file1)
    vocab2 = extract_vocab(vocab_file2)

    merged_vocab = vocab1 + [token for token in vocab2 if token not in vocab1]

    unique_vocab = []
    seen_tokens = set()
    for token in merged_vocab:
        if token not in seen_tokens:
            unique_vocab.append(token)
            seen_tokens.add(token)

    with open(merged_vocab_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(unique_vocab))

vocab_file1 = './bert_vocab/vocab.json'  
vocab_file2 = './new_vocab/vocab.json'  
merged_vocab_file = 'ex_vocab.txt'  

merge_vocabularies(vocab_file1, vocab_file2, merged_vocab_file)