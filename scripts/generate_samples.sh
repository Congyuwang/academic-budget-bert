PATH_TO_SHARDS=../wikidata/sharded
OUT_PATH=../wikidata/generated_samples
PATH_TO_VOCAB=../vocab/bert-large-uncased-vocab.txt
mkdir -p $OUT_PATH
python ../dataset/generate_samples.py\
	--dir $PATH_TO_SHARDS\
	-o $OUT_PATH\
	--dup_factor 10\
	--seed 42\
	--vocab_file $PATH_TO_VOCAB\
	--do_lower_case 1\
	--masked_lm_prob 0.15\
	--max_seq_length 128\
	--model_name bert-large-uncased\
	--max_predictions_per_seq 20\
	--n_processes 36
