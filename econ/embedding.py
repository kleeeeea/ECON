import json
import sys
from util.common import flatten, removeNonLetter, getLogger
from smart_open import smart_open
import itertools
from gensim.models import word2vec, Word2Vec
import random
import spacy
import logging
import re
import gensim
import argparse
import ipdb
logging.basicConfig(level=logging.DEBUG)

FROM_SUPERSEQUENCE = True
WINDOW_SIZE = 5
ITER = int(5)
# int(3e4) required for one sentence to converge..
REMOVE_CAPITAL = True
LEMMATIZE = False
ONLY_ENDS_WITH = False
# nlp = spacy.load('en')

dataset = 'test'

log = getLogger(__file__, stream_handler=None)


re_concept_tagged = re.compile(
    r"<c>(?P<phrase>[^<]*)</c>"
)


def any2unicode(text, encoding='utf8', errors='strict'):
    """Convert a string (bytestring in `encoding` or unicode), to unicode."""
    if isinstance(text, unicode):
        return text
    return unicode(text, encoding, errors=errors)


to_unicode = any2unicode


_nlp = spacy.load('en')

re_concept_tagged = re.compile(
    r"<c>(?P<phrase>[^<]*)</c>"
)


def to_concept_gensim(w):
    return '<c>%s</c>' % to_oneWord(w)


def is_concept_gensim(w):
    return re_concept_tagged.match(w)


def to_concept_natural_one_word(w):
    # return w
    return re.sub(r'</?c>', '', w)


def to_concept_natural(w):
    # return w
    return re.sub(r'</?c>', '', w).replace('_', ' ')


def to_concept_natural_lower(w):
    return to_concept_natural(w.lower())


def to_oneWord(w):
    return w.replace(' ', '_')


def word2internal(raw_textual_unit):
    try:
        if LEMMATIZE:
            doc = _nlp(raw_textual_unit)
            return '_'.join([w.lemma_ for w in doc])

        if REMOVE_CAPITAL:
            if not raw_textual_unit.istitle():
                raw_textual_unit = raw_textual_unit.lower()
        return raw_textual_unit.replace(' ', '_')
    except Exception as e:
        ipdb.set_trace()


def get_candidate_list(superspan):
    try:
        if superspan['tag'] == 'superspan':
            return [to_concept_gensim(span['text']) for span in superspan['spans']]
        else:
            return [superspan['text']]
        # try=======6345-09=====xt']]
        pass
    except Exception as e:
        ipdb.set_trace()


def getNormalizedTextualUnits(superspan):
    textual_units_raw = get_candidate_list(superspan)
    textual_units_normalized = [word2internal(raw_textual_unit) for raw_textual_unit in textual_units_raw]

    if ONLY_ENDS_WITH:
        return [span for span in textual_units_normalized if to_concept_natural_lower(superspan['text']).endswith(to_concept_natural_lower(span).split('_')[-1])]

    return textual_units_normalized


# def getNormalizedTextualUnits_preprocess(superspan):
#     textual_units_raw = get_candidate_list(superspan)
#     textual_units_normalized = [word2internal(raw_textual_unit) for raw_textual_unit in textual_units_raw]

#     return textual_units_normalized


def get_cleaned_superspan_sequence(superspan_sequence):
    superspan_sequence_removed_letters = [superspan for superspan in superspan_sequence if removeNonLetter(superspan['text'])]
    # if ONLY_ENDS_WITH:
    #     superspan_sequence_removed_letters_removed_nonEndingSpan = None

    return superspan_sequence_removed_letters


def get_list_of_candidateLists(superspan_sequence):
    superspan_sequence_removed_letters = [superspan for superspan in superspan_sequence if removeNonLetter(superspan['text'])]
    return [getNormalizedTextualUnits(superspan) for superspan in superspan_sequence_removed_letters]


# class LineSentenceAsWordPair(object):
#     """
#     Simple format: one sentence = one line; words already preprocessed and separated by whitespace.
#     """

#     def __init__(self, source, limit=None):
#         self.source = source
#         self.limit = limit

#     def __iter__(self):
#         with smart_open(self.source) as fin:
#             for line in itertools.islice(fin, self.limit):
#                 sentence = to_unicode(line).split()
#                 i = 0
#                 for i in range(len(sentence)):
#                     for j in range(i + 1, min(i + WINDOW_SIZE + 1, len(sentence))):
#                         # pairs_of_words.append([sentence[i], sentence[j]])
#                         yield [sentence[i], sentence[j]]


class LineSuperWordSequenceAsWordPair(object):
    def __init__(self, source, limit=None):
        self.source = source
        self.limit = limit

    def __iter__(self):
        with smart_open(self.source) as fin:
            for line in itertools.islice(fin, self.limit):
                superspan_sequence = get_list_of_candidateLists(json.loads(line))
                for i in range(len(superspan_sequence)):
                    for j in range(i + 1, min(i + WINDOW_SIZE + 1, len(superspan_sequence))):
                        for candidate_i in superspan_sequence[i]:
                            for candidate_j in superspan_sequence[j]:
                                yield [candidate_i, candidate_j]


# class LineSuperWordSequenceAsRandomSentence(object):
#     def __init__(self, source, limit=None):
#         self.source = source
#         self.limit = limit

#     def __iter__(self):
#         with smart_open(self.source) as fin:
#             for line in itertools.islice(fin, self.limit):
#                 superspan_sequence = get_list_of_candidateLists(json.loads(line))
#                 for i in range(len(superspan_sequence)):
#                     import ipdb; ipdb.set_trace()
#                     for j in range(i + 1, min(i + WINDOW_SIZE + 1, len(superspan_sequence))):
#                         for span_i in getNormalizedTextualUnits(superspan_sequence[i]):
#                             for span_j in getNormalizedTextualUnits(superspan_sequence[j]):
#                                 yield [span_i, span_j]

def trim_rule(word, count, min_count):
    if re_concept_tagged.match(word):
        return gensim.utils.RULE_KEEP
    return gensim.utils.RULE_DEFAULT


if __name__ == '__main__':

    parser = argparse.ArgumentParser("")
    parser.add_argument('arg1', nargs='?', default="cs", help="1st Positional arg")
    args = parser.parse_args()
    tokenized_text = args.arg1
    supersequence_path = tokenized_text + '_superspan_sequence.json'

    # output
    model_save_path = tokenized_text + '_embedding.bin'

    if FROM_SUPERSEQUENCE:
        file = supersequence_path
        model = Word2Vec(LineSuperWordSequenceAsWordPair(file), min_count=30, window=WINDOW_SIZE, sg=1, iter=ITER, workers=32, hs=1, negative=0, trim_rule=trim_rule)
    else:
        file = concept_representation_path
        model = Word2Vec(word2vec.LineSentence(file), min_count=5, window=WINDOW_SIZE, sg=1, iter=ITER, workers=32, hs=1, negative=0)

    model.save(model_save_path)
