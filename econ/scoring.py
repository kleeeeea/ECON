import csv
import re
from gensim.models import Word2Vec
import numpy as np
import pdb
from sklearn.svm import LinearSVC, SVC
from itertools import islice
from sklearn import preprocessing
import os
import argparse


parser = argparse.ArgumentParser("")
parser.add_argument('arg1', nargs='?', default="cs", help="1st Positional arg")
args = parser.parse_args()
tokenized_text = args.arg1
supersequence_path = tokenized_text + '_superspan_sequence.json'
model_save_path = supersequence_path + '_embedding.bin'
concept_feature_bin_path = tokenized_text + '_econ_feature.bin'
concept_feature_path = tokenized_text + '_econ_feature.txt'

inFile = concept_feature_path
model = Word2Vec.load(model_save_path)

dataset = 'machine_learning'
quality_phrase_file = '/scratch/home/hwzha/workspace/dbpedia/result/{}/phrase_list_without_lower.txt'.format(dataset)
modelFile = '/scratch/home/klee/workspace/conceptMining/data/{}/embedding_tmp.bin'.format(dataset)



def get_feature(row):
    try:
        if len(row) != 2:
            return
        text = row[1].strip()
        res = re.split('\s+', text[1:-1].strip())
        res = [float(r) for r in res]
        if len(res) == 4:
            return res
    except Exception as e:
        print(e)
        pdb.set_trace()


feature_dict = {}
with open(inFile) as fin:
    for i, line in enumerate(fin):
        row = line.split('\t')
        feature = get_feature(row)
        if feature:
            phrase = row[0]
            feature_dict[phrase] = feature
        else:
            break


SIZE = 1000

dbpedia_phrase_dict = {}
with open(dbpedia_phrase_file) as f_dbpedia:
    for line in f_dbpedia:
        phrase, freq = line.strip().split('\t')
        freq = int(freq)
        dbpedia_phrase_dict[phrase] = freq

neg_phrase = []
for phrase in islice(reversed(list(feature_dict.keys())), random.randint(0, list(feature_dict.keys())-SIZE), SIZE):
    if phrase in feature_dict:
        if np.random.random() > 0.5:
            neg_phrase.append(phrase)
        if len(neg_phrase) > 1000:
            break

# neg_phrase = []
# for phrase, freq in dbpedia_phrase_dict.items():
#     if freq > 1:
#         continue
#     neg_phrase.append()


pos_phrase = []
for phrase, freq in dbpedia_phrase_dict.items():
    if ' ' in phrase:
        phrase = phrase.replace(' ', '_')
        if phrase in feature_dict:
            pos_phrase.append(phrase)
        if len(pos_phrase) > SIZE:
            break

