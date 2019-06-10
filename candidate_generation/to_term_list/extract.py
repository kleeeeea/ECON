import argparse
from rake_nltk import Rake
from collections import defaultdict
from summa.keywords import keywords

from constants import METHOD_RAKE, METHOD_TEXTRANK

SUFFIX_TERM = '_extracted_terms.txt'


def method_name2suffix(method):
    return "_" + method + SUFFIX_TERM


r = Rake()


def get_keyword_list(file, method='textrank'):
    output_file = file + method_name2suffix(method)
    
    def get_termscoreList_from_text(extractedText):
        if method == 'textrank':
            return keywords(extractedText, scores=True)
        r.extract_keywords_from_text(extractedText)
        score_term_List = r.get_ranked_phrases_with_scores()
        return [(t[1], t[0]) for t in score_term_List]

    numLineReadEachExtraction = 5
    if method == 'textrank':
        numLineReadEachExtraction = 10
    f = open(file, "r")
    lineno = 0
    line = f.readline()
    lineno += 1
    termDict = defaultdict(int)
    lines = []

    while line:
        for i in range(0, numLineReadEachExtraction):
            if (line):
                lines.append(line)
                line = f.readline()
                lineno += 1
            else:
                break
        termScoreList = get_termscoreList_from_text(' '.join(lines))
        for termTuple in termScoreList:
            term, score = termTuple
            termDict[term] = termDict[term] + score * len(lines) / numLineReadEachExtraction
            # if term in termDict:
            #     if (score > termDict[term]):
            #         termDict[term] = score
            # else:
            #     termDict[term] = score
        lines.clear()
    f.close()

    termFile = open(output_file, "w")
    for term, score in sorted(termDict.items(), key=lambda x: x[1], reverse=True):
        if len(term.split(' ')) > 5:
            continue
        termFile.write(term)
        termFile.write("\t")
        termFile.write(str(score))
        termFile.write("\n")
    termFile.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("")
    parser.add_argument('arg1', nargs='?', default="cs", help="1st Positional arg")
    parser.add_argument('arg2', nargs='?', default=METHOD_RAKE, help="1st Positional arg")
    args = parser.parse_args()
    method = args.arg2

    inFile = args.arg1
    get_keyword_list(inFile, method)