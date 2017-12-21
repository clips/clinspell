from pyxdameraulevenshtein import damerau_levenshtein_distance
from doublemetaphone import dm
import json
import sys

#---------------------------------------------------------------------------
#                   EDIT DISTANCE SEARCH
#---------------------------------------------------------------------------


def load_vocab(vocab):
    """
    Transforms a vocabulary to the dictionary format required for the candidate generation.
    :param vocab: a list containing the vocabulary
    :return: vocab_dict
    """

    # TRANSFORM VOCABULARY TO DICTIONARY

    # initialize vocab word length keys and character set length keys

    vocab_dict = {}
    min_len = len(min(vocab, key=len))
    max_len = len(max(vocab, key=len))
    item_lens = range(min_len, max_len+1)

    for item in item_lens:
        vocab_dict[item] = {}
        for i in range(1, max_len+1):
            vocab_dict[item][i] = set()

    # fill vocab according to word length and character set length

    for word in vocab:
        vocab_dict[len(word)][len(set(word))].add(word)

    return vocab_dict

def levenshtein_candidates(word, vocab_dict, editdistance=2):
    """
    Generates candidates
    :param word: the misspelling for which to generate replacement candidates
    :param vocab_dict: the output of load_vocab()
    :param editdistance: the maximum Damerau-Levenshtein edit distance
    :return:
    """

    candidates = []

    word_len = len(word)
    set_len = len(set(word))

    if word_len <= 2:
        word_lengths = range(word_len, word_len + 1 + editdistance)
    else:
        word_lengths = range(word_len-editdistance, word_len+1+editdistance)

    if set_len-editdistance > 0:
        set_lengths = range(set_len-editdistance, set_len+1+editdistance)
    else:
        set_lengths = range(set_len, set_len + 1 + editdistance)

    selection = []

    for i in word_lengths:
        key = vocab_dict[i]
        for j in set_lengths:
            selection += key[j]

    for item in set(selection):
        if damerau_levenshtein_distance(word, item) <= editdistance:
            candidates.append(item)

    full_candidates = list(set(candidates))

    return full_candidates


#---------------------------------------------------------------------------
#               METAPHONE SEARCH (~ Aspell 'soundslike' suggestions)
#---------------------------------------------------------------------------

def load_metaphones(vocab):
    """
    :param vocab_file: either a list containing the vocabulary, or a text file which contains one lexical item per line
    :return: dictionary with mappings between Double Metaphone representations and corresponding lexical items
    """

    # MAKE METAPHONE-LEXICAL MAPPING

    metaphone_dict = {}
    for item in vocab:
        metaphones = dm(item)
        for metaphone in metaphones:
            if metaphone:
                try:
                    metaphone_dict[metaphone].append(item)
                except KeyError:
                    metaphone_dict[metaphone] = []
                    metaphone_dict[metaphone].append(item)

    return metaphone_dict

def convert_candidates(metaphone_candidates, detection, metaphone_dict):

    """
    :param candidates: replacement candidates
    :param detection: misspelling
    :param metaphone_dict: output of load_metaphones()
    :return: candidates converted from Double Metaphone representation to normal lexical representation
    """

    converted_candidates = []
    for i, candidate in enumerate(metaphone_candidates):
        for item in metaphone_dict[candidate]:
            if len(set(item).intersection(set(candidate))) >= 1: # have at least one character in common
                if damerau_levenshtein_distance(item, detection) <= 3:  # enough overlap
                    converted_candidates.append(item)

    return converted_candidates


#---------------------------------------------------------------------------
#                       COMPLETE ALGORITHM
#---------------------------------------------------------------------------


def candidates(misspellings, language='en'):

    vocab = json.load(open("lexicon_" + language + ".json", 'r'))
    vocab_dict = load_vocab(vocab)

    print(str(len(misspellings)) + ' misspellings to generate candidates for')

    candidates_list = []

    print("Generating Damerau-Levenshtein candidates")
    for i, misspelling in enumerate(misspellings):
        candidates_list.append(levenshtein_candidates(misspelling, vocab_dict, editdistance=2))

    print("Generating Double Metaphone candidates edit distance 1")
    metaphone_dict = load_metaphones(vocab)
    vocab_dict = load_vocab(list(metaphone_dict.keys()))
    metaphone_candidates = [levenshtein_candidates(dm(misspelling)[0], vocab_dict, editdistance=1)
                            for misspelling in misspellings]
    soundslike_candidates = [convert_candidates(candidates, detection, metaphone_dict) for
                             candidates, detection in zip(metaphone_candidates, misspellings)]
    candidates_list = [list(set(candidates1 + candidates2)) for candidates1, candidates2 in
                  zip(candidates_list, soundslike_candidates)]

    return candidates_list


