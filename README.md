# concept relevance


## Notes

This software requires [AutoPhrase](https://github.com/shangjingbo1226/AutoPhrase) to extract domain keyphrases. Please refer to AutoPhrase documentation for installation and usage.

## Requirements

We will take Ubuntu for example.

* python 2.7
```
$ sudo apt-get install python
```
* other python packages
```
$ sudo pip install -r requirements.txt
```

## Build
Build AutoPhrase by Makefile in the terminal.
```
$ cd SegPhrase
$ make
```

## Run
First, to fit our model to a corpus, run
```
$ bash ./train.sh 
```
and specify the input/output by changing the following variables at the begining of the script
* ```TEXT```: the input text file
* ```MODEL```: the path to store the model

Then, query a specific set of documents, run
```
$ bash ./test.sh 
```
and specify the input/output by changing the following variables at the begining of the script
* ```TEXT```: the input text file
* ```CATEGORY_SEEDCONCEPTS```: one or more set of concepts to query
* ```MODEL```: the path for the stored model you wish to use
* ```SEGGED_TEXT_categorized```: the final output

## Input Format
* The input files specified by ```TEXT``` for both ```train.sh``` and ```test.sh``` should be one document per line. 
* The input file ```CATEGORY_SEEDCONCEPTS``` should have each line following the format ```[category name]\t[concept1],[concept2],[concept3]...```, and can contain one or more lines.

## Output Format
The query relevance output, as specified by the `SEGGED_TEXT_categorized` in the begining of the ```test.sh```, is of the format of [relevance to concept set1], [relevance to concept set2]...


## Hyper-Parameters
The running parameters are located in `conf.d` folder, including `autoPhrase.conf` and `pyConfig.conf`. 

`autoPhrase.conf` contains the parameters for concept extraction and segmentation.

`pyConfig.conf` contains the parameters for all other training steps

### segphrase.conf

```
MIN_SUP=10
```

A hard threshold of raw frequency is specified for frequent phrase mining, which
will generate a candidate set.

```
HIGHLIGHT_MULTI=0.5
```

The threshold for multi-word phrases to be recognized as quality phrases.


```
HIGHLIGHT_SINGLE=0.5
```

The threshold for multi-word phrases to be recognized as quality phrases.

### learning_embedding.conf

```
USE_CONCEPT_GRAPH=1
```
whether to use the concept graph distance algorithm (when set to `1`) or to use the basic query expansion algorithm (when set to `0`) to compute relevance.

```
MIN_NEIGHBOR_SIMILARITY=.6
```
minimum threshold for concept pairs to be considered neighbors in the concept graph

```
MIN_CATEGORY_NEIGHBOR=3
```
minimum number of neighbors that each concept in the query concept set should have

```
MAX_NEIGHBORS=100
```
maximum number of neighbors a concept in the concept will have
