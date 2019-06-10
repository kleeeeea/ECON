import ipdb
from tqdm import tqdm
import re
import json
import os
from multiprocessing import Pool
import sys
from pathos.multiprocessing import ProcessingPool as Pool
import argparse
import multiprocessing

import spotlight
from spotlight import SpotlightException
from requests import HTTPError
from constants import SUFFIX_DBPEDIA


def get_offset_to_index_dict(text):
    token_offset_to_index = {0: 0}
    index = 0
    for offset, w in enumerate(text):
        if w == ' ':
            index += 1
            token_offset_to_index[offset+1] = index
    return token_offset_to_index




def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def get_line_count(inFile):
    count = -1
    for count, line in enumerate(open(inFile, 'r')):
        pass
    count += 1
    return count


def validate_nps(nps, tokens):
    for np in nps:
        st = np['st']
        ed = np['ed']
        token_span = tokens[st:ed]
        np_span = np['text'].split(' ')
        if token_span != np_span:
            print('token span', token_span, 'np_span', np_span)
            return False
    return True


def dbpedia_extract_spans(line):
    validate = True
    threshold = 0.5
    text = line.strip()
    nps = []
    tokens = text.split(' ')
    try:
        token_offset_to_index = get_offset_to_index_dict(text)
        annotations = spotlight.annotate('http://localhost:2222/rest/annotate', line, confidence=threshold)
        for annotation in annotations:
            offset = annotation['offset']
            surfaceForm = annotation['surfaceForm']
            spaceNum = len(re.findall(' ', surfaceForm))
            try:
                st = token_offset_to_index[offset]
                ed = st + spaceNum + 1
                span = {'st': st, 'ed': ed, 'text': surfaceForm}
                if ' '.join(tokens[st:ed]) == surfaceForm:
                    nps.append(span)
            except Exception as e:
                pass
                # print(e)
                # ipdb.set_trace();
        if validate:
            if not validate_nps(nps, tokens):
                pass
                ipdb.set_trace()
    except (SpotlightException, HTTPError) as e:
        pass
    except Exception as e:
        print(e)
        pass
        # ipdb.set_trace();
    return nps


# def writeToJson(inFile, outFile, validate=False):
if __name__ == '__main__':

    '''
    [{'URI': 'http://dbpedia.org/resource/Indium',
      'offset': 0,
      'percentageOfSecondRank': 0.21560966948687654,
      'similarityScore': 0.8070463567236783,
      'support': 443,
      'surfaceForm': 'In',
      'types': ''}]
    '''

    parser = argparse.ArgumentParser("")
    parser.add_argument('arg1', nargs='?', default="cs", help="1st Positional arg")
    parser.add_argument('arg2', nargs='?', default="", help="1st Positional arg")
    args = parser.parse_args()

    inFile = args.arg1
    outFile = inFile + SUFFIX_DBPEDIA


    lineCount = get_line_count(inFile)
    with open(inFile) as fin:
        texts = [line for line in fin]

    with open(inFile) as fin, open(outFile, 'w') as fout:
        # p = Pool(10)
        # Pool(10)
        for line in tqdm(fin, total=lineCount):
            # for lines in tqdm(batch(texts, 10)):

            nps = dbpedia_extract_spans(line)
            # nps_pool = p.map(dbpedia_extract_spans, lines)
            # import ipdb; ipdb.set_trace()
            # for nps in nps_pool:
            fout.write(json.dumps(nps))
            fout.write('\n')


# if __name__ == '__main__':
#     writeToJson(inFile, outFile, True)
    # print("finish generating {} json file".format(dataset))
