IN_PATH=../labordata/extract/
OUT_PATH=../labordata/sharded
mkdir -p $OUT_PATH
python ../dataset/shard_data.py\
	--dir $IN_PATH\
	-o $OUT_PATH\
	--num_train_shards 256\
	--num_test_shards 128\
	--frac_test 0.1

