import sys
import spacy
import json
from tqdm import tqdm
from spacy.tokens import Doc
from candidate_generation.to_json.nltk_extract import validate_nps
from util.common import get_line_count, getLogger
import os
import argparse

SUFFIX_SPACY = '_spacy.json'

log = getLogger(os.path.basename(__file__))



try:
    unicode
except Exception as e:
    # python 3
    unicode = str


class WhitespaceTokenizer_spacy(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(' ')
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)


_NLP = spacy.load('en')
_NLP_backup = spacy.load('en')
_NLP.tokenizer = WhitespaceTokenizer_spacy(_NLP.vocab)


def get_nps_spacy(text, withEntity=True, withNounChunks=True, validate=True):
    try:
        doc = _NLP(text)

    except Exception as e:
        doc = _NLP_backup(text)

    nps = []

    if withEntity:
        for ent in doc.ents:
            np = {'st': ent.start, 'ed': ent.end, 'label': ent.label_, 'text': ent.text}
            nps.append(np)

    if withNounChunks:
        for np in doc.noun_chunks:
            nounphrase = {'st': np.start, 'ed': np.end, 'text': np.text}
            nps.append(nounphrase)

    if validate:
        words_original = text.split(' ')
        nps_v = validate_nps(nps, words_original)
        if nps_v != nps:
            log.info('words_original: {}'.format(words_original))
            log.info('nps: {}'.format(nps))
            return nps_v
    return nps


def writeToJson(inFile, outFilet):
    global _NLP
    with open(inFile, 'r') as fin, open(outFile, 'w') as fout:
        total = get_line_count(inFile)

        for i, line in tqdm(enumerate(fin), total=total):
            text = unicode(line).strip('\r\n')

            if text:
                nps, entities = get_nps_spacy(text, withEntity=False, withNounChunks=True), get_nps_spacy(text, withEntity=True, withNounChunks=False)
            else:
                nps = []
                entities = []

            fout.write(json.dumps({'entity': entities, 'np': nps}))
            fout.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser("")
    parser.add_argument('arg1', nargs='?', default="cs", help="1st Positional arg")
    args = parser.parse_args()
    inFile = args.arg1
    outFile = inFile + SUFFIX_SPACY

    writeToJson(inFile, outFile)
