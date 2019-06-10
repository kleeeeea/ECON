import sys
import nltk
import json
from tqdm import tqdm
import multiprocessing
import argparse
import os
from util.common import get_line_count, getLogger

SUFFIX_NLTK = '_nltk.json'

log = getLogger(os.path.basename(__file__))

VALIDATE = True



# get minimal
# get maximal
# then run with embedding and score them

# generate both linear and overlapping repr

def get_nps_from_tree(tree, words_original, attachNP=False, skip_single_word=True):
    nps = []
    st = 0
    for subtree in tree:
        if isinstance(subtree, nltk.tree.Tree):
            if subtree.label() == 'NP':
                np = subtree.leaves()
                ed = st + len(np)
                if not skip_single_word or len(np) > 1:
                    nps.append({'st': st, 'ed': ed,
                                'text': ' '.join(words_original[st:ed])})
                    if attachNP:
                        nps[-1]['np'] = np

            st += len(subtree.leaves())
        else:
            st += 1
    return nps


def validate_nps(nps, words_original):
    validated_nps = []
    for np in sorted(nps, key=lambda x:x['st']):
        st = np['st']
        ed = np['ed']
        token_span = words_original[st:ed]
        # 'A polynomial time algorithm for the Lambek calculus with brackets of  bounded order'
        if ' '.join(token_span).strip() != np['text'].strip():
            print(' '.join(token_span))
            print(np)
            return validated_nps
        validated_nps.append(np)
    return nps

GRAMMAR = r"""
NBAR:
  {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns
NP:
  {<NBAR>}
  {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...
"""
_PARSER = nltk.RegexpParser(GRAMMAR)  # chunk parser


def get_nps_nltk_raw(doc, validate=VALIDATE):
    words_original = doc.split(' ')
    try:
        parse_tree = _PARSER.parse(nltk.pos_tag(words_original))
        pass
    except Exception as e:
        import ipdb; ipdb.set_trace()
        pass
    nps = get_nps_from_tree(parse_tree, words_original)
    if validate:
        nps_v = validate_nps(nps, words_original)
        if nps_v != nps:
            log.info('words_original: {}'.format(words_original))
            log.info('nps: {}'.format(nps))
            return nps_v
    return nps

batch_size = multiprocessing.cpu_count() * 2
_pool = multiprocessing.Pool(batch_size)


def get_nps_nltk(doc, validate=VALIDATE):
    if type(doc) is list:
        return _pool.map(get_nps_nltk_raw, doc)
    else:
        return get_nps_nltk_raw(doc)


def writeToJson(inFile, outFile):
    with open(inFile, 'r') as fin, open(outFile, 'w') as fout:
        total = get_line_count(inFile)
        for line in tqdm(fin, total=total):
            doc = line.strip('\r\n')
            if doc:
                nps = get_nps_nltk(doc)
            else:
                nps = []

            fout.write(json.dumps(nps))
            fout.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser("")
    parser.add_argument('arg1', nargs='?', default="cs", help="1st Positional arg")
    args = parser.parse_args()
    inFile = args.arg1
    outFile = inFile + SUFFIX_NLTK

    writeToJson(inFile, outFile)
