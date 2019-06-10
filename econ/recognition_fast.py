import json
import sys
from gensim.models import word2vec, Word2Vec
import numpy as np
from econ.embedding import get_cleaned_superspan_sequence, WINDOW_SIZE, getNormalizedTextualUnits, re_concept_tagged, to_concept_natural_lower
from collections import deque
from collections import defaultdict
import pickle
import logging
import re
from itertools import groupby
import argparse
# context dominance
# e.g. range correlations  1

logging.basicConfig(level=logging.DEBUG)

ALPHA = .5
BETA = .51
GAMMA = 10
IS_DOMINATED_COEFF = -20


parser = argparse.ArgumentParser("")
parser.add_argument('arg1', nargs='?', default="cs", help="1st Positional arg")
args = parser.parse_args()
tokenized_text = args.arg1


MAX_CHOICES = 5000

BASIC_THRESHOLD = .5
restrict_vocab = 100000
TOPN = 30

supersequence_path = tokenized_text + '_superspan_sequence.json'
model_save_path = tokenized_text + '_embedding.bin'

# output
concept_representation_path = tokenized_text + '_concept_representation.txt'


model = Word2Vec.load(model_save_path)
concept_list = [w for w in model.wv.index2word if re_concept_tagged.match(w)]

concept_lowered2score = {}
concept_score_path = tokenized_text + '_score_list.bin'

try:
    concept_score_list = pickle.load(open(concept_score_path, 'r'))
    concept2score = dict(zip(concept_list[:len(concept_score_list)], concept_score_list))
    concept_lowered2score = {c.lower(): max([s for c, s in c_s]) for c, c_s in
                             groupby(sorted(concept2score.items(), key=lambda t: t[0]), key=lambda t: t[0])}
except FileNotFoundError as e:
    pass


vocab_lower = {k.lower():v for k,v in model.wv.vocab.items()}
concept_lower2Concept = {w:model.wv.index2word[vocab_lower[w].index] for w in vocab_lower}

# if we want to use
def score(current_spansChoice):
    return model.score([current_spansChoice])[0]


# def highest_scored(score_backtrace_candidates):
#     return sorted(score_backtrace_candidates, key=lambda x:x[0], reverse=True)[0]

def getNormalizedLengthScore(sequence, superspan_sequence):
    score = sum([len(to_concept_natural_lower(w).split('_')) / float(len(superspan['text'].split())) for w, superspan in zip(sequence, superspan_sequence)])
    return score


def getConceptQualityScore(sequence):
    score = sum([concept_lowered2score.get(w.lower(), 0) for w in sequence])
    # todo: add length rewards
    return score


def getEndsWithScore(sequence, superspan_sequence):
    score = sum([1 if to_concept_natural_lower(superspan['text']).endswith(to_concept_natural_lower(span).split('_')[-1]) else 0 for span, superspan in zip(sequence, superspan_sequence)])
    return score


def getIsDominatedScore(sequence, superspan_sequence, model=model):
    score = 0
    for concept, superspan in zip(sequence, superspan_sequence):
        if re_concept_tagged.match(concept):
            concept = concept.lower()
            # if '<c>positive_definite_matrix' in concept:
            #     import ipdb; ipdb.set_trace()
            covered_concepts = set()
            try:
                c1overed_neighbor_word2sim = {covered_concept.lower(): sim for covered_concept, sim in model.most_similar(model.wv.index2word[vocab_lower[concept1].index], topn=TOPN, restrict_vocab=restrict_vocab, partition_only=True) if sim > BASIC_THRESHOLD and to_concept_natural_lower(concept) in to_concept_natural_lower(covered_concept)}
                for other_concept in getNormalizedTextualUnits(superspan):
                    if other_concept.lower() in covered_neighbor_word2sim:
                        covered_concepts.add(other_concept)
                        continue

                score += len(covered_concepts)
            except Exception as e:
                continue



            # if len(covered_concepts) > 0:
            #     import ipdb; ipdb.set_trace()

    return score


def normalize(array):
    if not isinstance(array, np.ndarray):
        array = np.array(array, dtype=np.float)
    return (array - np.min(array)) / (np.max(array) - np.min(array) + np.finfo(float).eps)


# [model, quality, length, endswith]
def select_best(superspan_sequence, ALPHA = .5, BETA = .51, GAMMA = 10, model=model):
    # combine all choices and score each one, select best
    possible_sequence_bylength = defaultdict(list)
    for span in getNormalizedTextualUnits(superspan_sequence[0]):
        possible_sequence_bylength[0] += [[span]]
    for i in range(1, len(superspan_sequence)):
        current_superSpan = superspan_sequence[i]
        # todo: take in non overlapping textual units within same superspan
        for span in getNormalizedTextualUnits(current_superSpan):
            possible_sequence_bylength[i] += [previous_possible_sequence + [span] for previous_possible_sequence in possible_sequence_bylength[i-1]]
    possible_sequences = possible_sequence_bylength[len(superspan_sequence) - 1]
    model_scores = normalize(model.score(possible_sequences))

    # is add, because is computing negative log likelihood

    # if two words are similar in the same span, the contained words are dominated and will not be selected
    concept_quality_scores = normalize([getConceptQualityScore(possible_sequence) for possible_sequence in possible_sequences])
    concept_length_scores = normalize([getNormalizedLengthScore(possible_sequence, superspan_sequence) for possible_sequence in possible_sequences])
    concept_endswith_scores = normalize([getEndsWithScore(possible_sequence, superspan_sequence) for possible_sequence in possible_sequences])
    concept_is_dominated_scores = normalize([getIsDominatedScore(possible_sequence, superspan_sequence, model) for possible_sequence in possible_sequences])
    try:
        scores = model_scores + ALPHA * concept_quality_scores + BETA * concept_length_scores + GAMMA * concept_endswith_scores + IS_DOMINATED_COEFF * concept_is_dominated_scores
    except Exception as e:
        import ipdb; ipdb.set_trace()

    original_sent = ' '.join(superspan['text'] for superspan in superspan_sequence)
    print('\n'.join(['%s %s %s %s %s %s %s' % t for t in zip(possible_sequences, model_scores, concept_quality_scores, concept_length_scores, concept_endswith_scores, concept_is_dominated_scores, scores)]))
    print(original_sent)
    print(sorted(zip(possible_sequences, scores), key=lambda x: x[1], reverse=True)[0][0])
    # import ipdb; ipdb.set_trace()
    return sorted(zip(possible_sequences, scores), key=lambda x: x[1], reverse=True)[0][0]


# todo: add combination of sequence

def process_superspan_sequence(superspan_sequence, ALPHA=.5, BETA=.51, GAMMA=10, model=model):
    superspan_sequence = get_cleaned_superspan_sequence(superspan_sequence)
    recognized_spans = []

    current_start = 0
    num_current_choices = 1

    for i, current_superSpan in enumerate(superspan_sequence):
        # compute max. choice
        num_current_choices *= len(getNormalizedTextualUnits(current_superSpan))

        # if MAX_CHOICES is reached
        if num_current_choices >= MAX_CHOICES:
            # score all sentence, merge into result
            recognized_spans += select_best(superspan_sequence[current_start:i + 1], ALPHA=ALPHA, BETA=BETA, GAMMA=GAMMA, model=model)
            current_start = i + 1
            num_current_choices = 1

    if superspan_sequence[current_start:]:
        recognized_spans += select_best(superspan_sequence[current_start:], model=model)

    return ' '.join(recognized_spans)


def process_all():
    with open(concept_representation_path, 'w') as f_out:
        for l in open(supersequence_path):
            concept_representation = process_superspan_sequence(json.loads(l))
            # import ipdb; ipdb.set_trace()
            # logging.debug('selected %s' % concept_representation)
            f_out.write(concept_representation.strip() + '\n')


with open(supersequence_path) as fin:
    supersequences = [i for i in fin]


from tqdm import tqdm


def process_by_index(indexes=None):
    indexes = indexes or range(len(supersequences))

    model.init_sims()

    with open(concept_representation_path, 'w') as f:
        for i in tqdm(indexes):
            segmentation = process_superspan_sequence(json.loads(supersequences[i]), model=model)
            f.write(segmentation + '\n')

if __name__ == '__main__':
    process_by_index(range(5))