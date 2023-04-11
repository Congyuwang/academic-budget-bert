pip install wikiextractor
IN_PATH=../wikidata/enwiki-20230401-pages-articles.xml.bz2
OUT_PATH=../wikidata/extract
mkdir -p $OUT_PATH
wikiextractor -o $OUT_PATH --processes 30 $IN_PATH
