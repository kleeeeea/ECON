
# coding: utf-8

# In[1]:


from operator import itemgetter
import os
from tqdm import tqdm


# In[2]:


workspaceDir = '/scratch/home/hwzha/workspace'
dataset = 'test'
phraseinFile = '{workspaceDir}/AutoPhrase/models/{dataset}/AutoPhrase.txt'.format(
    workspaceDir=workspaceDir, dataset=dataset)


# def read_autophrase_list(inFile, threshold=0.5):
# #     for test
# #     inFile = phraseinFile
# #     threshold = 0.5
#     with open(inFile, 'r') as fin:
#         phrase_list = []
#         for line in fin:
#             phrase = line.split('\t')[1].strip()
#             score = float(line.split('\t')[0])
#             phrase_list.append((phrase, score))
#             if score < threshold:
#                 break
#     return phrase_list

# def read_rake_list(inFile, threshold, phraseFirst=True):
#     with open(inFile, 'r') as fin:
#         phrase_list = []
#         for line in fin:
#             phrase = line.split('\t')[0].strip()
#             score = float(line.split('\t')[1])
#             if score >= threshold:
#                 phrase_list.append((phrase, score))
#         return phrase_list


# method = 'textrank'
# phraseinFile = '{workspaceDir}/{method}/{dataset}_{method}_term.txt'.format(
#     workspaceDir=workspaceDir, method=method, dataset=dataset)
# phrase_list = read_rake_list(phraseinFile, 0)
# phrase_list_sorted = sorted(phrase_list, key=itemgetter(1), reverse=True)

# phrase_list_sorted_ = []
# for phrase in phrase_list_sorted:
#     phrase_len = len(phrase[0].split(' '))
#     if phrase_len <= 6 and phrase_len > 1:
#         phrase_list_sorted_.append(phrase)

# phraseinFile = '{workspaceDir}/AutoPhrase/models/{dataset}/AutoPhrase.txt'.format(
#     workspaceDir=workspaceDir, dataset=dataset)
# phrase_list = read_autophrase_list(phraseinFile, 0.5)
# phrase_list[:10]


# In[4]:


def read_phrase_list(inFile, threshold, phraseFirst=True, min_phrase_len=2, max_phrase_len=6):
    if phraseFirst:
        phrase_idx = 0
        score_idx = 1
    else:
        phrase_idx = 1
        score_idx = 0
    try:
        with open(inFile, 'r') as fin:
            phrase_list = []
            for line in fin:
                phrase = line.split('\t')[phrase_idx].strip()
                score = float(line.split('\t')[score_idx])
                phrase_list.append((phrase, score))

            phrase_list_sorted = sorted(phrase_list, key=itemgetter(1), reverse=True)
            phrase_list_sorted_ = []
            for phrase in phrase_list_sorted:
                phrase_len = len(phrase[0].split(' '))
                if phrase_len <= max_phrase_len and phrase_len >= min_phrase_len:
                    phrase_list_sorted_.append(phrase)
            return phrase_list_sorted_
    except FileNotFoundError as e:
        return None


# In[5]:


def save_phrase_list(phrase_list, outFile):
    if phrase_list:
        outDir = os.path.dirname(os.path.realpath(outFile))
        if not os.path.exists(outDir):
            os.makedirs(outDir)
        with open(outFile, 'w') as fout:
            for p in phrase_list:
                fout.write(p[0] + '\t' + str(p[1]))
                fout.write('\n')


# In[6]:


for dataset in tqdm(['test', 'nips', 'pubmed', 'JMLR', 'database']):
    method = 'auto'
    phraseinFile = '{workspaceDir}/AutoPhrase/models/{dataset}/AutoPhrase.txt'.format(
        workspaceDir=workspaceDir, method=method, dataset=dataset)
    phrase_list = read_phrase_list(phraseinFile, threshold=0.0, phraseFirst=False,min_phrase_len=2, max_phrase_len=6)
    # phrase_list[:10]
    phraseoutFile = '{workspaceDir}/{method}/result/{dataset}/phrase_list.txt'.format(
        workspaceDir=workspaceDir, method=method, dataset=dataset)
    save_phrase_list(phrase_list, phraseoutFile)

    method = 'textrank'
    phraseinFile = '{workspaceDir}/{method}/{dataset}_{method}_term.txt'.format(
        workspaceDir=workspaceDir, method=method, dataset=dataset)
    phrase_list = read_phrase_list(phraseinFile, threshold=0.0, phraseFirst=True,min_phrase_len=2, max_phrase_len=6)
    # phrase_list[:10]
    phraseoutFile = '{workspaceDir}/{method}/result/{dataset}/phrase_list.txt'.format(
        workspaceDir=workspaceDir, method=method, dataset=dataset)
    save_phrase_list(phrase_list, phraseoutFile)

    method = 'rake'
    phraseinFile = '{workspaceDir}/{method}/{dataset}_{method}_term0.txt'.format(
        workspaceDir=workspaceDir, method=method, dataset=dataset)
    phrase_list = read_phrase_list(phraseinFile, threshold=0.0, phraseFirst=True,min_phrase_len=2, max_phrase_len=6)
#     phrase_list[:10]
    phraseoutFile = '{workspaceDir}/{method}/result/{dataset}/phrase_list.txt'.format(
        workspaceDir=workspaceDir, method=method, dataset=dataset)
    save_phrase_list(phrase_list, phraseoutFile)

    method = 'kea'
    phraseinFile = '{workspaceDir}/{method}/{dataset}_{method}_term.txt'.format(
        workspaceDir=workspaceDir, method=method, dataset=dataset)
    phrase_list = read_phrase_list(phraseinFile, threshold=0.0, phraseFirst=True,min_phrase_len=2, max_phrase_len=6)
    # phrase_list[:10]
    phraseoutFile = '{workspaceDir}/{method}/result/{dataset}/phrase_list.txt'.format(
        workspaceDir=workspaceDir, method=method, dataset=dataset)
    save_phrase_list(phrase_list, phraseoutFile)

