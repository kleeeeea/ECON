import os

import argparse
import sys
import json
from tqdm import tqdm
import ipdb
import re

from constants import SUFFIX_AUTOPHRASE
from candidate_generation.to_json.nltk_extract import validate_nps
import subprocess
from util.common import get_line_count, condenseSpace

AUTOPHRASE_PATH = os.environ.get('AUTOPHRASE_PATH')


def model2segmented_text_path(MODEL):
    return os.path.join(AUTOPHRASE_PATH, MODEL, "segmentation.txt")


def train_autophrase(text_to_seg, model):
    if os.path.exists(model2segmented_text_path(model)):
        # caching based on segmented_text_path
        return

    os.environ['RAW_TRAIN'] =  os.path.abspath(text_to_seg)
    os.environ['MODEL'] = model
    mycwd = os.getcwd()
    os.chdir(AUTOPHRASE_PATH)
    proc = subprocess.Popen('bash auto_phrase.sh'.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in proc.stdout:
        print(line)
    os.chdir(mycwd)


def segment(text_to_seg, model):
    os.environ['TEXT_TO_SEG'] = os.path.abspath(text_to_seg)   # tell autophrase with abspath
    os.environ['MODEL'] = model
    mycwd = os.getcwd()
    os.chdir(AUTOPHRASE_PATH)
    proc = subprocess.Popen('bash phrasal_segmentation.sh'.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in proc.stdout:
        print(line)
    os.chdir(mycwd)


def removeMarker(text):
    # '<phrase>'
    return re.sub('</?phrase>', '', text)


# [\w ]+</phrase>

# def validate_nps(nps, tokens):
#     for np in nps:
#         st = np['st']
#         ed = np['ed']
#         token_span = tokens[st:ed]
#         np_span = np['text'].split(' ')
#         if token_span != np_span:
#             print('token span', token_span, 'np_span', np_span)
#             return False
#     return True


def writeToJson(inFile, outFile, originalFile):
    with open(inFile, 'r') as fin, open(outFile, 'w') as fout, open(originalFile, 'r') as fOriginal:
        # with open(inFile, 'r') as fin:
        total = get_line_count(inFile)

        cnt = 0
        data = []
        for i, (line, line_original) in tqdm(enumerate(zip(fin, fOriginal)), total=total):
            text = line.strip()

            tokens = text.split(' ')
            original_tokens = line_original.split()
            clean_tokens = condenseSpace(removeMarker(text)).split(' ')
            nps = []
            for idx, token in enumerate(tokens):
                if '<phrase>' in token:
                    if token.startswith('<phrase>'):
                        span = {'st': idx}
                    else:
                        span = {}
                elif '</phrase>' in token:
                    if token.endswith('</phrase>'):
                        try:
                            if span:
                                span['ed'] = idx + 1
                                span['text'] = ' '.join(clean_tokens[span['st']:span['ed']])
                                nps.append(span)
                            span = {}
                        except Exception as e:
                            ipdb.set_trace()
                            print(e)
                    else:
                        span = {}
            if nps:
                nps_v = validate_nps(nps, original_tokens)
                if nps_v != nps:
                    ipdb.set_trace()
                nps = nps_v
            fout.write(json.dumps(nps))
            fout.write('\n')

            # fout.write(json.dumps(nps))
            # fout.write('\n')


# os.path.basename(tokenized_text)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("")
    parser.add_argument('arg1', nargs='?', default="cs", help="1st Positional arg")
    parser.add_argument('--model', default="", help="model name for Autophrase")
    args = parser.parse_args()

    inFile = args.arg1
    tokenized_text_autophrase = inFile + SUFFIX_AUTOPHRASE

    # use the base path of tokenized_text as
    model = args.model or os.path.basename(inFile)

    if model == 'pubmed_cleaned':
        segment(inFile, 'pubmed')
    else:
        print('train_autophrase' + '=' * 60)
        train_autophrase(inFile, model)
        print('segment' + '=' * 60)
        segment(inFile, args.arg1)
    writeToJson(model2segmented_text_path(model), tokenized_text_autophrase, inFile)

    # inFile = sys.argv[1]
    # outFile = sys.argv[2]
    # # inFile = "/scratch/home/hwzha/workspace/AutoPhrase/models/test/segmentation.txt"
    # # outFile = "/scratch/home/hwzha/workspace/auto/result/test/merged.txt_without_sentence_id.json"
