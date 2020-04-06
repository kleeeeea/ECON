#!/usr/bin/env bash
RAW_TEXT='sample_data/sample_text.txt'

#install autophrase according to https://github.com/shangjingbo1226/AutoPhrase and change this
export AUTOPHRASE_PATH="$HOME/bin/AutoPhrase"

#install dbpedia according to https://github.com/dbpedia-spotlight/spotlight-docker and change this
export DBPEDIA_PATH="$HOME/bin/spotlight-english"
docker run -i -p 2222:80 dbpedia/spotlight-english spotlight.sh

# Out-of-the-box: download best-matching default model and create shortcut link
unalias python
python -m spacy download en
python -m spacy download en_core_web_sm

#install dbpedia spotlight at   https://github.com/dbpedia-spotlight/spotlight-docker

python candidate_generation/to_json/nltk_extract.py $RAW_TEXT
python candidate_generation/to_json/spacy_extract.py $RAW_TEXT
python candidate_generation/to_json/autophrase.py $RAW_TEXT
python candidate_generation/to_json/dbpedia_extract.py $RAW_TEXT

python candidate_generation/to_term_list/extract.py $RAW_TEXT textrank
python candidate_generation/to_term_list/extract.py $RAW_TEXT rake
python candidate_generation/to_term_list/seg_with_vocab.py $RAW_TEXT textrank
python candidate_generation/to_term_list/seg_with_vocab.py $RAW_TEXT rake

python candidate_generation/merge_span.py $RAW_TEXT

#follow the procedure in https://github.com/kleeeeea/ECON/blob/master/notebooks/classifier.ipynb to generate the score list file

python econ/embedding.py $RAW_TEXT
python econ/scoring_feature_generation.py $RAW_TEXT

# run the notebook classifier.ipynb to obtain scored concepts
python econ/recognition_fast.py $RAW_TEXT
