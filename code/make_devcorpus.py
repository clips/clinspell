# dependencies
from pyxdameraulevenshtein import damerau_levenshtein_distance

# built-in packages
import random
import json
import string
import sys

# Damerau-Levenshtein operations

def letterswitch(word):
    if len(word) > 2:
        i = random.randint(0, len(word)-2)
    elif len(word) == 2:
        i = 0
    else:
        raise ValueError("Word input must be longer than 1 character")
    letterlist = list(word)
    letterlist[i], letterlist[i+1] = letterlist[i+1], letterlist[i]

    return "".join(letterlist)

def letterdelete(word):
    i = random.randint(1, len(word)-1)  # avoid omitting first letter, rarely happens
    letterlist = list(word)
    letterlist.pop(i)
    return "".join(letterlist)

def letterinsert(word):
    i = random.randint(1, len(word))
    letterlist = list(word)
    letterlist.insert(i, random.choice(string.ascii_lowercase))
    return "".join(letterlist)

def lettersub(word):
    i = random.randint(1, len(word)-1)
    letterlist = list(word)
    letterlist[i] = random.choice(string.ascii_lowercase.replace(letterlist[i], ""))
    return "".join(letterlist)


# Corpus sample function
def corpus_sample(corpus, samplename, samplesize):
    """
    :param corpus: preprocessed corpus text file to sample from
    :param samplename: name of the output sample
    :param samplesize: number of lines to sample from the corpus
    :return: list of sampled lines from corpus
    """

    with open(corpus, 'r') as f:

        sample_lines = []

        # random sample of defined samplesize
        len_lines = 0
        for line in f:
            len_lines += 1
        sample = set(random.sample(range(len_lines), samplesize))

    with open(corpus, 'r') as f:

        for i, line in enumerate(f):
            if i in sample:
                print("Yes!")
                sample_lines.append(line)

    with open(samplename, 'w') as f:
        json.dump((sample_lines, sorted(list(sample))[::-1]), f)

    print("Done")


# Devcorpus creation function

def make_devcorpus(corpusfile, language, outfile, window_size=10, oov=False, samplesize=0, editdistance=12):
    """
    :param corpusfile: file containing the corpus to sample from
    :param lexicon_file: json file containing a reference lexicon
    :param pathtovectors: path to vec file containing trained fastText vectors
    :param oov: True if the distorted words are absent from the vector vocabulary
    :param samplesize: number of lines to sample
    :param editdistance: the type of edit distances generated: 1, 2 or 1 and 2 (80-20 proportion)
    :return: devcorpus with all relevant lists
    """

    # load lexicon
    assert language in ['en', 'nl']
    with open('lexicon_' + language + '.json', 'r') as f:
        vocab = set(json.load(f))

    # load vector vocab
    with open('../data/embeddings_' + language + '.vec', 'r') as f:
        vector_vocab = set([line.strip() for i, line in enumerate(f) if i > 0])

    # load sample
    if samplesize:
        corpus_sample(corpusfile, 'devsample.json', samplesize)
        with open('devsample.json', 'r') as f:
            corpus = json.load(f)[0]
    else:
        with open(corpusfile, 'r') as f:
            corpus = json.load(f)[0]

    # generate misspellings corpus with corrections and detection contexts
    functionlist = [letterswitch, letterdelete, letterinsert, lettersub]
    correct_spellings = []
    misspellings = []
    misspelling_contexts = []
    used_samplelines = []  # to be able to backtrack
    distance_idxs = [] # keep track which element has which edit distance

    for j, line in enumerate(corpus):
        print('Line {}:'.format(j))
        if len(line.split()) < 1 + window_size * 2:
            # check whether a context frame of min_window_size tokens on each side can be extracted for at least 1 token
            print('Line too short, skipping to next line.')
            continue
        print('Collecting candidate words...')
        idxs = []
        for i, word in enumerate(line.split()):
            if not oov:
                if (word in vocab) and (len(word) > 3) and (word in vector_vocab):
                    if (i-window_size >= 0) and (i + window_size < len(line.split())):
                        # check whether a context frame of min_window_size tokens on each side can be extracted
                        idxs.append(i)
            else:
                if (word in vocab) and (len(word) > 3) and (word not in vector_vocab):
                    if (i-window_size >= 0) and (i + window_size < len(line.split())):
                        idxs.append(i)
        if idxs:
            i = random.choice(idxs)
            word = line.split()[i]
            print("{} candidates, sampled '{}' ".format(len(idxs), word))
        else:  # skip to next line if no eligible words
            print('No eligible words found, skipping to next line.')
            continue

        correct_spellings.append(word)
        misspelling = ''

        if editdistance == 1:
            REDLIGHT = 1
            while REDLIGHT:
                misspelling = random.choice(functionlist)(word)
                if misspelling not in vocab:
                    misspellings.append(misspelling)
                    distance_idxs.append(1)
                    REDLIGHT = 0

        elif editdistance == 2:
            REDLIGHT = 1
            while REDLIGHT:
                misspelling = random.choice(functionlist)(random.choice(functionlist)(word))
                if (misspelling not in vocab) and (damerau_levenshtein_distance(misspelling, word) == 2):
                    misspellings.append(misspelling)
                    distance_idxs.append(2)
                    REDLIGHT = 0

        elif editdistance == 12:  # 80% edit distance 1, 20% edit distance 2
            if j % 5 == 0:
                REDLIGHT = 1
                while REDLIGHT:
                    misspelling = random.choice(functionlist)(random.choice(functionlist)(word))
                    if (misspelling not in vocab) and (damerau_levenshtein_distance(misspelling, word) == 2):
                        misspellings.append(misspelling)
                        distance_idxs.append(2)
                        REDLIGHT = 0
            else:
                REDLIGHT = 1
                while REDLIGHT:
                    misspelling = random.choice(functionlist)(word)
                    if misspelling not in vocab:
                        misspellings.append(misspelling)
                        distance_idxs.append(1)
                        REDLIGHT = 0

        # append corresponding context window
        misspelling_contexts.append(((" ".join(line.split()[i-window_size:i])),
                                     (" ".join(line.split()[i+1:i+window_size+1]))))

        # append index of used line in sample
        used_samplelines.append(j)

        # keeping track
        print('Resulting misspelling: {}'.format(misspelling))
        print("{} instances collected".format(len(misspellings)))

    print('Finished corpus generation. Saving...')
    # save corpus
    with open(outfile, 'w') as f:
        json.dump((correct_spellings, misspellings, misspelling_contexts, used_samplelines, distance_idxs), f)
    print('Done.')

    # return corpus
    return correct_spellings, misspellings, misspelling_contexts, used_samplelines, distance_idxs


# Sample items from generated devcorpus to get perfect 80-20 balance between 1 and 2 Damerau-Levenshtein distance

def sample_80_20(devcorpusfile, amount=0):
    """
    :param devcorpusfile: generated devcorpus
    :param amount: amount of instances to retain, if not specified it retains the maximum possible amount
    :return: balanced devcorpus
    """

    with open(devcorpusfile, 'r') as f:
        devcorpus = json.load(f)

    correct_spellings = devcorpus[0]
    misspellings = devcorpus[1]
    detection_contexts = devcorpus[2]
    used_samplelines = devcorpus[3]
    distance_idxs = devcorpus[4]

    print('Original amount of instances:')
    print(len(correct_spellings))

    if amount:

        idxs_1_distance = [i for i, id in enumerate(distance_idxs) if id == 1]
        idxs_2_distance = [i for i, id in enumerate(distance_idxs) if id == 2]

        proportions = [(amount * 4) // 5, amount // 5]

        filtered_1_idxs = random.sample(idxs_1_distance, proportions[0])
        filtered_2_idxs = random.sample(idxs_2_distance, proportions[1])
        all_idxs = sorted(list(set(filtered_1_idxs + filtered_2_idxs)))

    else:

        idxs_1_distance = [i for i, id in enumerate(distance_idxs) if id == 1]
        idxs_2_distance = [i for i, id in enumerate(distance_idxs) if id == 2]

        while len(idxs_1_distance) < len(idxs_2_distance)*4:
            idxs_1_distance.pop(random.choice(range(len(idxs_1_distance))))

        all_idxs = sorted(list(set(idxs_1_distance + idxs_2_distance)))

    print('Amount of instances after balancing:')
    print(len(all_idxs))

    correct_spellings = [x for i, x in enumerate(correct_spellings) if i in all_idxs]
    misspellings = [x for i, x in enumerate(misspellings) if i in all_idxs]
    detection_contexts = [x for i, x in enumerate(detection_contexts) if i in all_idxs]
    used_samplelines = [x for i, x in enumerate(used_samplelines) if i in all_idxs]
    distance_idxs = [x for i, x in enumerate(distance_idxs) if i in all_idxs]

    devcorpus = [correct_spellings, misspellings, detection_contexts, used_samplelines, distance_idxs]

    with open(devcorpusfile, 'w') as f:
        json.dump(devcorpus, f)

if __name__ == "__main__":
    make_devcorpus(sys.argv[1], sys.argv[2], sys.argv[3], window_size=sys.argv[4],
                   oov=eval(sys.argv[5]), samplesize=int(sys.argv[6]))
    sample_80_20(sys.argv[3])