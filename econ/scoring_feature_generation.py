import sys
import argparse
from gensim.models import word2vec, Word2Vec
import numpy as np
import re
import pickle
import logging
from econ.embedding import to_concept_natural, re_concept_tagged, to_concept_gensim, to_concept_natural_lower
from util.common import removeNonLetter, getLogger


parser = argparse.ArgumentParser("")
parser.add_argument('arg1', nargs='?', default="cs", help="1st Positional arg")
args = parser.parse_args()
tokenized_text = args.arg1
supersequence_path = tokenized_text + '_superspan_sequence.json'
model_save_path = supersequence_path + '_embedding.bin'

# output
concept_feature_bin_path = tokenized_text + '_econ_feature.bin'

concept_feature_path = tokenized_text + '_econ_feature.txt'


dataset = args.arg1

model = Word2Vec.load(model_save_path)


# dataset2BASIC_THRESHOLD = {
#     'pubmed': .5,
#     'database': .3,
#     'machine_learning': .5,
# }
# BASIC_THRESHOLD = dataset2BASIC_THRESHOLD[dataset]

# dataset2restrict_vocab = {
#     'pubmed': 100000,
#     'database': 50000,
#     'machine_learning': 200000,
# }
# restrict_vocab = dataset2restrict_vocab[dataset]

# dataset2TOPN = {
#     'pubmed': 20,
#     'database': 20,
#     'machine_learning': 20,
# }
# TOPN = dataset2TOPN[dataset]
BASIC_THRESHOLD = .5
restrict_vocab = None
TOPN = 50


VALIDATION_SIZE = 20
import random


def validate_model(model, model_save_path='', include_score=True):
    valid_window = min(100, len(model.wv.index2word))
    validation_size = min(VALIDATION_SIZE, len(model.wv.index2word))
    top_k = 10

    # if not model_save_path:
    #     model_save_path = '/tmp/model_%s' % datetime.datetime.now()
    # model.save(model_save_path)

    print(model.wv.index2word[:100])

    valid_examples_frequent = random.sample(model.wv.index2word, validation_size // 2)
    valid_examples_phrase = random.sample([word for word in model.wv.index2word if re_concept_tagged.match(word)], validation_size // 2)

    try:
        model['analysis']
        valid_examples_frequent[0] = 'analysis'
        model[to_concept_gensim( 'machine_learning')]
        valid_examples_phrase[0] = to_concept_gensim( 'machine_learning')
    except Exception as e:
        pass

    for valid_word in valid_examples_frequent + valid_examples_phrase:
        print('valid word %s' % to_concept_natural(valid_word))
        if include_score:
            print('%s' % [(to_concept_natural(word), score) for word, score in model.most_similar(valid_word)])
        else:
            print('%s' % [to_concept_natural(word) for word, score in model.most_similar(valid_word)])


index2word_normalize_conceptd = [to_concept_natural_lower(w) for w in model.wv.index2word]
index2word_normalize_conceptd_reverse = {w: i for i, w in enumerate(index2word_normalize_conceptd)}

re_nonASCII = re.compile(r'[^\x00-\x7F]+')
wiki_concept_set = set()

import os
AUTOPHRASE_PATH = os.environ.get('AUTOPHRASE_PATH')

# load target_word_set
for l in open(os.path.join(AUTOPHRASE_PATH, 'data/EN/wiki_quality.txt')):
    concept = l.strip()
    if not re_nonASCII.search(l):
        wiki_concept_set.add(to_concept_natural_lower(concept).replace(' ', '_'))

target_concept_set = wiki_concept_set

try:
    dbpedia_concept_set = set()
    for l in open("/scratch/home/hwzha/workspace/%s/result/%s/phrase_list.txt" % ('dbpedia', dataset)):
        concept, score = l.strip().split('\t')
        if len(concept.split(' ')) > 1 or concept.isupper():
            dbpedia_concept_set.add(to_concept_natural_lower(concept).replace(' ', '_'))

    target_concept_set |= dbpedia_concept_set

except Exception as e:
    pass

# try:
#     from gensim.similarities.index import AnnoyIndexer
#     model.init_sims()
#     annoy_index = AnnoyIndexer(model, 100)
#     indexer = annoy_index
# except Exception as e:
#     indexer = None
target_concept_index_set = sorted([index2word_normalize_conceptd_reverse[w] for w in target_concept_set & set(index2word_normalize_conceptd)])
restrict_vocab = None

concept_feature_dict = {}


def computeFeatures(concept, model):
    neighbor_word2sim = {to_concept_natural_lower(w): sim for w, sim in model.most_similar(concept, topn=TOPN, restrict_vocab=restrict_vocab) if sim > BASIC_THRESHOLD}
    if len(neighbor_word2sim) < 2:
        return np.array([0, 0, 0, 0])

    # Meaningfulness: no. neighbors
    meaningfulness = len(neighbor_word2sim)
    # Purity:avg. similarity
    purity = np.mean(list(neighbor_word2sim.values()))

    # Targetness:no. known words
    targetness = len(set(neighbor_word2sim.keys()) & target_concept_set)
    # {to_concept_natural_lower(w): sim for w, sim in model.most_similar(concept, topn=3, limited_index=target_concept_index_set) if sim > BASIC_THRESHOLD}
    # len(set(neighbor_word2sim.keys()) & target_concept_set)c

    completeness = -len([w for w in neighbor_word2sim.keys() if to_concept_natural_lower(concept) in w])
    # contained_by_set = [w for w in index2word_normalize_conceptd if w.endswith(to_concept_natural_lower(concept))]
    # contained_by_index_set = [index2word_normalize_conceptd_reverse[w] for w in contained_by_set]
    # # Completeness:do not have phrase that contains it
    # completeness = {to_concept_natural_lower(w): sim for w, sim in model.most_similar(concept, topn=3, limited_index=contained_by_index_set) if sim > BASIC_THRESHOLD}

    return np.array([meaningfulness, purity, targetness, completeness])


def generate_score():
    with open(concept_feature_path, 'w') as f_out:
        for i, w in enumerate(model.wv.index2word):
            if i % 10000 == 0:
                logging.debug('%sth concept' % i)
                pickle.dump(concept_feature_dict, open(concept_feature_bin_path, 'wb'))
            if re_concept_tagged.match(w):
                concept_feature_dict[w] = computeFeatures(w, model)
                f_out.write('%s\t%s\n' % (to_concept_natural(w), concept_feature_dict[w]))

    # concept_feature_matrx = np.array([concept_feature_dict[w] for w in model.wv.index2word])


def scoring():
	pass

if __name__ == '__main__':
    generate_score()
