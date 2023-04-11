pip install wikiextractor
IN_PATH=../wikidata/enwiki-20230401-pages-articles.xml.bz2
OUT_PATH=../wikidata/extract
mkdir -p $OUT_PATH
python ../dataset/process_data.py -f $IN_PATH --n_processes 30 -o $OUT_PATH --type wiki
