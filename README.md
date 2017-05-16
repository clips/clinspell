This repository contains source code for the paper 'Unsupervised Context-Sensitive Spelling Correction for English and Dutch Clinical Free-Text with Word and Character N-gram Embeddings', which is currently under review for [CLIN Journal](http://www.clinjournal.org). The source code offered here contains scripts to extract our manually annotated MIMIC-III data, and to
run the experiments described in the paper.

# License

MIT

# Requirements

* Python 3
* Numpy
* [Reach](https://github.com/stephantul/reach)
* [pyxdameraulevenshtein](https://github.com/gfairchild/pyxDamerauLevenshtein)
* [Facebook fastText](https://github.com/facebookresearch/fastText)
* [fasttext](https://github.com/salestock/fastText.py), a Python interface for Facebook fastText

All requirements are available from pip, except ```fastText```. To install these requirements, just run

```pip install -r requirements.txt```

from inside the cloned repository.

In order to build ```fastText```, use the following:

```
$ git clone https://github.com/facebookresearch/fastText.git
$ cd fastText
$ make
```

To extract our manually annotated MIMIC-III test data, you should have access to the [MIMIC-III database](https://mimic.physionet.org).

# Usage

## Extracting the English test data

To extract the annotated test data, git clone this repository and place the file **NOTEEVENTS.csv** from the MIMIC-III database inside the data directory of this repository. 
Then run 

```python3 extract_test.py```

This script preprocesses the **NOTEEVENTS.csv** data and stores the preprocessed data in the file **mimic_preprocessed.txt**. It then extracts the annotated 
test data, which is stored to the file **testcorpus.json** in four lists: correct replacements, misspellings, misspelling contexts, and line indices.

## Extracting development data and other resources

### Preprocessing

To generate development corpora as described in the paper, the data has to be preprocessed. To preprocess English data, run

```python3 preprocess.py [path to raw data] [path to created preprocessed data]```

This script uses the source code of the English tokenizer from [Pattern](https://github.com/clips/pattern). 

To preprocess Dutch data, you can use the [Ucto](https://languagemachines.github.io/ucto/) tokenizer and, for every line, retain every token which 
matches 

```r'(^[^\d\W])[^\d\W]*(-[^\d\W]*)*([^\d\W]$)'```

### Generating frequency lists and neural embeddings

To extract a frequency list from the preprocessed data, run

```python3 frequencies.py [path to preprocessed data] [language]```

The [language] argument should always either be **en** if the language is English or **nl** if the language is Dutch. 

To train the fastText vectors as we do, place the preprocessed data in the cloned fastText directory and run

```./fasttext skipgram -input [path to preprocessed data] -output ../data/embeddings_[language] -dim 300```

This makes an embeddings_[language].vec and embeddings_[language].bin file in the data repository.

### Generating development corpora

To create a development corpus from preprocessed data, run

```python3 make_devcorpus.py [path to preprocessed data] [language] [path to created devcorpus] [allow_oov] [samplesize]```

The [allow_oov] argument should be False for development setup 1 or 2 from the paper, and True for development setup 3. 
The [samplesize] argument should contain the number of lines to sample from the data.

## Conducting experiments

### Generating candidates

To generate candidates for a created development corpus, run

```python3 candidates.py [path to preprocessed data] 2 [name of output] [language]```

To generate candidates for our extracted test data or other empirically observed data, run

```python3 candidates.py [path to preprocessed data] all [name of output] [language]```

### Ranking experiments

The ```Development``` class in **ranking_experiments.py** contains all functions to conduct the experiments. 

Example:

```
import ranking_experiments

# load devcorpus for setup 1, 2 and 3

with open('devcorpus_setup1.json', 'r') as f:
        corpusfiles_setup1 = json.load(f)
devcorpus_setup1 = [corpusfiles[0], corpusfiles[1], corpusfiles[2]]

with open('devcorpus_setup2.json', 'r') as f:
        corpusfiles_setup2 = json.load(f)
devcorpus_setup2 = [corpusfiles[0], corpusfiles[1], corpusfiles[2]]

with open('devcorpus_setup3.json', 'r') as f:
        corpusfiles_setup3 = json.load(f)
devcorpus_setup3 = [corpusfiles[0], corpusfiles[1], corpusfiles[2]]

# load candidates for setup 1, 2 and 3
with open('candidates_devcorpus_setup1.json', 'r') as f:
        candidates_setup1 = json.load(f)
with open('candidates_devcorpus_setup2.json', 'r') as f:
        candidates_setup2 = json.load(f)
with open('candidates_devcorpus_setup3.json', 'r') as f:
        candidates_setup3 = json.load(f)

# perform grid search
scores_setup1 = Development.grid_search(devcorpus_setup1, candidates_setup1, language='en')
scores_setup2 = Development.grid_search(devcorpus_setup2, candidates_setup2, language='en')

# search for best averaged parameters
best_parameters = Development.define_best_parameters('iv'=[scores_setup1, scores_setup2])

# perform grid search for oov penalty
oov_scores_setup1 = Development.tune_oov(devcorpus_setup1, candidates_list, best_parameters, language='en')
oov_scores_setup2 = Development.tune_oov(devcorpus_setup2, candidates_list, best_parameters, language='en')
oov_scores_setup3 = Development.tune_oov(devcorpus_setup3, candidates_list, best_parameters, language='en')

# search for best averaged oov penalty
best_oov = Development.define_best_parameters('iv'=[oov_scores_setup1, oov_scores_setup2], 'oov'=oov_scores_setup3)

# store best parameters
best_parameters['oov_penalty'] = best_oov
with open('parameters.json', 'w') as f:
	json.dump(best_parameters, f)

# conduct ranking experiments with best parameters on test data

with open('testcorpus.json', 'r') as f:
	testfiles = json.load(f)
testcorpus = [testfiles[0], testfiles[1], testfiles[2]]

with open('testcandidates.json', 'r') as f:
        testcandidates = json.load(f)

# ranking experiment and analysis per frequency scenario for our context-sensitive model, noisy channel model, and majority frequency

best_parameters['ranking_method'] = 'context'
dev = Development(best_parameters, language='en')
accuracy_context, correction_list_context = dev.conduct_experiment(testcorpus, testcandidates)
frequency_analysis_context = dev.frequency_analysis()

best_parameters['ranking_method'] = 'noisy_channel'
dev = Development(best_parameters, language='en')
accuracy_noisychannel, correction_list_noisychannel = dev.conduct_experiment(testcorpus, testcandidates)
frequency_analysis_noisychannel = dev.frequency_analysis()

best_parameters['ranking_method'] = 'frequency'
dev = Development(best_parameters, language='en')
accuracy_frequency, correction_list_frequency = dev.conduct_experiment(testcorpus, testcandidates)
```












