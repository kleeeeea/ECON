import sys
import spacy
import json
from tqdm import tqdm
import sys
from candidate_generation.to_json.nltk_extract import validate_nps
from candidate_generation.to_term_list.extract import METHOD_TEXTRANK, METHOD_RAKE, method_name2suffix as method_name2suffix_termlist
from spacy.tokens import Doc
import argparse

from spacy.matcher import PhraseMatcher

try:
    unicode
except Exception as e:
    # python 3
    unicode = str


VALIDATE = True
THRESHOLD = 0.0

def method_name2suffix(method):
    return "_" + method + '.json'

class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(' ')
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)


def get_line_count(inFile):
  count = -1
  for count, line in enumerate(open(inFile, 'r')):
     pass
  count += 1
  return count


nlp = spacy.load('en', disable=['ner', 'parser', 'tagger'])
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)


def read_phrase_list(inFile, threshold=0.5):
  # unified phrase file format
  # input is already ranked and cut off
  # runtime complexity \t 0.7805276116
  with open(inFile, 'r') as fin:
    phrase_list = []
    for line in fin:
        phrase = line.split('\t')[0].strip()
        score = float(line.split('\t')[1])
        if score < threshold:
            break
        phrase_list.append(phrase)
    return phrase_list


def writeToJson(inFile, outFile, matcher, validate=False):
  global nlp
  total = get_line_count(inFile)
  with open(inFile, 'r') as fin, open(outFile, 'w') as fout:

    for line in tqdm(fin, total=total):
      text = unicode(line).strip('\r\n')
      tokens = text.split(' ')

      if text:
        # text may be empty when line is \n
        doc = nlp(text.lower())
        matches = matcher(doc)
        '''
        matches:
        [(1826470356240629538, 0, 1),
         (4451351154198579052, 0, 2),
         (7342778914265824300, 1, 2),
         (3411606890003347522, 2, 3)]
        '''
        nps = []
        for m in matches:
          st = m[1]
          ed = m[2]
          np = {'st':st, 'ed':ed, 'text': ' '.join(tokens[st:ed])}
          nps.append(np)
      else:
        nps = []

      if validate:
        nps = validate_nps(nps, tokens)

      fout.write(json.dumps(nps))
      fout.write('\n')


def extract_by_phrase_list(inFile, phraseinFile, outFile):
    with open(phraseinFile, 'r') as fin:
      phrase_list = read_phrase_list(phraseinFile, THRESHOLD)
    matcher = PhraseMatcher(nlp.vocab)
    for v in phrase_list:
      matcher.add(v, None, nlp(unicode(v)))  # python3 is unicode
    writeToJson(inFile, outFile, matcher, validate=VALIDATE)

import os

if __name__ == '__main__':

    # if not threshold
    #   threshold = 0.0
    parser = argparse.ArgumentParser("")
    parser.add_argument('arg1', nargs='?', default="cs", help="1st Positional arg")
    parser.add_argument('arg2', nargs='?', default=METHOD_RAKE, help="1st Positional arg")
    args = parser.parse_args()
    method = args.arg2

    inFile = args.arg1

    phraseinFile = inFile + method_name2suffix_termlist(method)

    if os.path.exists(phraseinFile):
        outFile = inFile + method_name2suffix(method)
        extract_by_phrase_list(inFile, phraseinFile, outFile)
