# dependencies
from pyxdameraulevenshtein import damerau_levenshtein_distance
import fasttext
from doublemetaphone import dm
import numpy as np

# built-in packages
from functools import reduce
from statistics import mean
from math import log
import random
import json
import time

# development experiments


class Development(object):

    def __init__(self, parameters, language):

        assert language in ["en", "nl"]
        self.language = language

        # load frequency list
        pathtofrequencies = 'frequencies_' + language + '.json'
        # load trained fasttext model
        pathtomodel = 'embeddings_' + language + '.bin'

        # PHASE 1
        self.comp_function = parameters['comp_function']  # item from ["sum", "mult", "max"]
        self.include_misspelling = parameters['include_misspelling']  # boolean
        self.include_oov_candidates = parameters['include_oov_candidates']  # boolean
        self.model = fasttext.load_model(pathtomodel)   # path to fasttext model

        # PHASE 2
        self.window_size = parameters['window_size']  # number in range(0,11)
        self.reciprocal = parameters['reciprocal']  # boolean
        self.remove_stopwords = parameters['remove_stopwords']  # boolean
        self.stopwords = frozenset(json.load(open('stopwords_' + str(self.language) + '.json', 'r')))

        # PHASE 3
        self.edit_distance = parameters['edit_distance']  # item from [1, 2, 3, 4]

        # PHASE 4
        self.oov_penalty = parameters['oov_penalty']  # oov penalty tuned with self.tune_oov()

        # OUTPUT
        self.ranking_method = parameters['ranking_method']  # item from ["context", "noisy_channel", "frequency",
        # "ensemble"]
        self.frequency_dict = json.load(open(pathtofrequencies, 'r'))  # path to frequency list
        self.k = parameters['k-best']  # positive natural number

    @staticmethod
    def comp_sum(vectors, reciprocal=False):
        """
        :param vectors: vectors to be composed
        :param reciprocal: if True, apply reciprocal weighting
        :return: composed vector representation
        """
        if not reciprocal:
            composed_vector = np.sum(vectors, axis=0)
        else:
            weight_vector = np.reciprocal(np.arange(1., len(vectors) + 1))
            weighted_vectors = []
            for i, weight in enumerate(weight_vector):
                weighted_vectors.append(vectors[i] * weight)
            composed_vector = np.sum(weighted_vectors, axis=0)

        return composed_vector

    @staticmethod
    def comp_mult(vectors, reciprocal=False):
        """
        :param vectors: vectors to be composed
        :param reciprocal: if True, apply reciprocal weighting
        :return: composed vector representation
        """
        if not reciprocal:
            composed_vector = reduce(lambda x, y: x * y, vectors)
        else:
            weight_vector = np.reciprocal(np.arange(1., len(vectors) + 1))
            weighted_vectors = []
            for i, weight in enumerate(weight_vector):
                weighted_vectors.append(vectors[i] * weight)
            composed_vector = reduce(lambda x, y: x * y, weighted_vectors)

        return composed_vector

    @staticmethod
    def comp_max(vectors, reciprocal=False):
        """
        :param vectors: vectors to be composed
        :param reciprocal: if True, apply reciprocal weighting
        :return: composed vector representation
        """
        if not reciprocal:
            composed_vector = np.amax(vectors, axis=0)
        else:
            weight_vector = np.reciprocal(np.arange(1., len(vectors) + 1))
            weighted_vectors = []
            for i, weight in enumerate(weight_vector):
                weighted_vectors.append(vectors[i] * weight)
            composed_vector = np.amax(weighted_vectors, axis=0)

        return composed_vector

    @staticmethod
    def normalize(vector):
        """
        Normalizes a vector.
        :param vector: a numpy array or list to normalize.
        :return a normalized vector.
        """
        if not vector.any():
            return vector

        return vector / np.linalg.norm(vector)

    def vectorize(self, sequence, remove_oov=True):
        """
        :param sequence: sequence to be vectorized
        :param remove_oov: whether to vectorize oov tokens
        :return: vectorized sequence
        """
        if remove_oov:
            sequence = [x for x in sequence if x in self.model.words]

        return [np.array(self.model[x]) for x in sequence]

    @staticmethod
    def spell_score(misspelling, candidates, method=1):
        """
        Calculates the edit distance between a misspelling and each candidate according to the chosen method
        :param misspelling: misspelling
        :param candidates: list of candidates
        :param method: chosen method from [1, 2, 3, 4]
        :return: list of edit distances between misspelling and each candidate
        """
        lexical_scores = [damerau_levenshtein_distance(misspelling, candidate)
                          for candidate in candidates]

        if method == 1:
            return lexical_scores
        else:
            phonetic_scores = [damerau_levenshtein_distance(dm(misspelling)[0], dm(candidate)[0])
                               for candidate in candidates]

        if method == 2:
            return [phonetic_score if phonetic_score != 0 else 1 for phonetic_score in phonetic_scores]
        elif method == 3:
            return [0.5 * (a + b) for a, b in zip(lexical_scores, phonetic_scores)]
        elif method == 4:
            return [(2 * a + b) ** 2 for a, b in zip(lexical_scores, phonetic_scores)]
        else:
            raise ValueError('Method must be element from [1, 2, 3, 4]')

    def ranking_experiment(self, detection_list, detection_contexts, candidates_list):
        """
        Experimental implementation of our context-sensitive ranking model.
        :param detection_list: list of misspellings
        :param detection_contexts: list of misspelling context tuples ('left context', 'right context')
        :param candidates_list: list of candidate list per misspelling
        :param r: loaded vector representations
        :return: list with corrections or k-best corrections
        """
        correction_list = []

        for misspelling, context, candidates in zip(detection_list, detection_contexts, candidates_list):

            # PHASE 1 AND 2: composition method and context weighting

            processed_context = ['', '']
            processed_context[0] = " ".join(context[0].split()[::-1][:self.window_size])
            processed_context[1] = " ".join(context[1].split()[:self.window_size])

            comp = self.comp_function

            if comp == "sum":
                comp_function = self.comp_sum
            elif comp == "mult":
                comp_function = self.comp_mult
            else:
                comp_function = self.comp_max

            if self.remove_stopwords:
                processed_context[0] = [t for t in processed_context[0] if t not in self.stopwords]
                processed_context[1] = [t for t in processed_context[1] if t not in self.stopwords]

            center = self.normalize(np.array(self.model[misspelling]))  # create or call vector representation for misspelling
            left_window = self.vectorize(processed_context[0], remove_oov=True)  # take only in-voc tokens
            right_window = self.vectorize(processed_context[1], remove_oov=True)  # take only in-voc tokens

            if left_window:
                vectorized_left_window = comp_function(left_window, reciprocal=self.reciprocal)
            else:
                vectorized_left_window = np.zeros(len(self.model.dim))

            if right_window:
                vectorized_right_window = comp_function(right_window, reciprocal=self.reciprocal)
            else:
                vectorized_right_window = np.zeros(len(self.model.dim))

            if self.include_misspelling:
                vectorized_context = comp_function((vectorized_left_window, center, vectorized_right_window))
            else:
                vectorized_context = comp_function((vectorized_left_window, vectorized_right_window))
            vectorized_context = self.normalize(vectorized_context)

            candidate_vectors = []
            remove_idxs = []
            oov_idxs = []

            # make vector representations of candidates
            for i, candidate in enumerate(candidates):
                if candidate in self.model.words:
                    candidate_vectors.append(self.normalize(np.array(self.model[candidate])))
                else:
                    if self.include_oov_candidates:
                        candidate_vectors.append(self.normalize(np.array(self.model[candidate])))
                        oov_idxs.append(i)
                    else:
                        remove_idxs.append(i)

            # update candidate list
            candidates = [candidate for i, candidate in enumerate(candidates) if i not in remove_idxs]

            # calculate cosine similarities
            distances = [np.dot(vectorized_context, candidate) for candidate in candidate_vectors]

            # PHASE 3: edit distance penalty

            method = self.edit_distance
            if method:
                spell_scores = self.spell_score(misspelling, candidates, method=method)
                distances = [a / b for a, b in zip(distances, spell_scores)]

            # PHASE 4: oov criteria
            if self.include_oov_candidates:
                for i, d in enumerate(distances):
                    if i in oov_idxs:
                        distances[i] /= self.oov_penalty

            # OUTPUT
            if self.k == 1:
                try:
                    correction_list.append(candidates[np.argmax(distances)])
                except ValueError:
                    correction_list.append('')
            elif self.k > 1:
                correction_list.append([candidates[i] for i in np.argsort(distances)[::-1][:self.k]])
            else:
                raise ValueError('k must be positive natural number')

        return correction_list

    def noisychannel_ranking(self, detection_list, candidates_list):
        """
        An approximate implementation of the ranking method described in (Lai et al. 2015)
        :param detection_list: list of misspellings
        :param candidates_list: list of candidate list per misspelling
        :param frequency_dict: corpus frequencies from training data
        :param k_best: if True, return k highest ranked candidates instead of single one
        :return: list with corrections or k-best corrections
        """

        correction_list = []
        confidences = []

        for misspelling, candidates in zip(detection_list, candidates_list):
            score_list = []
            for candidate in candidates:
                orthographic_edit_distance = damerau_levenshtein_distance(misspelling, candidate)
                phonetic_edit_distance = damerau_levenshtein_distance(dm(misspelling)[0], dm(candidate)[0])

                spell_score = (2 * orthographic_edit_distance + phonetic_edit_distance) ** 2  # P(m|c)

                try:
                    frequency = self.frequency_dict[candidate]
                except KeyError:
                    frequency = 1

                frequency_score = 1 / (1 + log(frequency))  # P(c)

                score = spell_score * frequency_score  # P(c|m) = P(m|c)*P(c)
                score_list.append(score)

            score_list = np.array(score_list)

            if len(score_list) > 1:
                sorted_distances = [score_list[i] for i in np.argsort(score_list)]
                top1 = sorted_distances[0]
                top2 = sorted_distances[1]
                confidence = abs(top1 - top2) / top1
                confidences.append(confidence)
            else:
                confidences.append(0)

            if self.k == 1:
                try:
                    correction_list.append(candidates[np.argmin(score_list)])
                except ValueError:
                    correction_list.append('')
            elif self.k > 1:
                correction_list.append([candidates[i] for i in np.argsort(score_list)[:self.k]])
            else:
                raise ValueError('k must be positive natural number')

        self.confidences = confidences

        return correction_list

    def frequency_baseline(self, detection_list, candidates_list):
        """
        Majority frequency baseline
        :param detection_list: list of misspellings
        :param candidates_list: list of candidate list per misspelling
        :return: list with corrections or k-best corrections
        """
        correction_list = []

        for misspelling, candidates in zip(detection_list, candidates_list):

            candidates = [candidate for candidate in candidates if candidate in self.frequency_dict.keys()]

            frequencies = [self.frequency_dict[candidate] for candidate in candidates]

            if self.k == 1:
                try:
                    correction_list.append(candidates[np.argmax(frequencies)])
                except ValueError:
                    correction_list.append('')
            elif self.k > 1:
                correction_list.append([candidates[i] for i in np.argsort(frequencies)[::-1][:self.k]])
            else:
                raise ValueError('k must be positive natural number')

        return correction_list

    @staticmethod
    def sub_sampling(correction_list, corrected_list, k=10):
        """
        Calculates the correction accuracy averaged over k subsampled folds
        :param correction_list: list of corrections
        :param corrected_list: list of gold standard corrections
        :param k: number of folds
        :return: correction accuracy averaged over k subsampled folds
        """
        length = len(correction_list)
        all_idxs = list(range(length))
        random.seed(0.56)
        random.shuffle(all_idxs)

        folds_length = length // k

        heldout_parts = [all_idxs[folds_length*i:folds_length*(i+1)] for i in range(k)]

        scores = []

        for heldout_part in heldout_parts:

            test_idxs = [i for i in all_idxs if i not in heldout_part]

            corrects = 0
            for i in test_idxs:
                if correction_list[i] == corrected_list[i]:
                    corrects += 1

            score = corrects/len(test_idxs)
            scores.append(score)

        return mean(scores)

    def conduct_experiment(self, devcorpus, candidates_list):
        """
        Streamlines experiments with the various ranking modules
        :param devcorpus: devcorpus generated with make_devcorpus.py
        :param candidates_list: list of candidate list per misspelling
        :return: correction accuracy, list of corrections
        """
        corrected_list = devcorpus[0]
        detection_list = devcorpus[1]
        detection_contexts = devcorpus[2]

        self.corrected_list = corrected_list
        self.detection_list = detection_list
        self.detection_contexts = detection_contexts
        self.candidates_list = candidates_list

        if self.ranking_method == 'context':
            correction_list = self.ranking_experiment(detection_list, detection_contexts, candidates_list)
        elif self.ranking_method == 'noisy_channel':
            correction_list = self.noisychannel_ranking(detection_list, candidates_list)
        elif self.ranking_method == 'frequency':
            correction_list = self.frequency_baseline(detection_list, candidates_list)
        elif self.ranking_method == 'ensemble':
            correction_list = self.ranking_experiment(detection_list, detection_contexts, candidates_list)
            correction_list_2 = self.noisychannel_ranking(detection_list, candidates_list)
            for i, confidence in enumerate(self.confidences):
                if confidence > 1.3:
                    correction_list[i] = correction_list_2[i]
        else:
            raise ValueError('No valid ranking method given')

        score = self.sub_sampling(correction_list, corrected_list)

        self.correction_list = correction_list
        self.score = score

        return score, correction_list

    @staticmethod
    def grid_search(devcorpus, candidates_list, language):
        """
        Conduct grid search to find best parameters for a corpus containing only in-vector-vocabulary corrections
        :param devcorpus: devcorpus generated with make_devcorpus.py
        :param candidates_list: list of candidate list per misspelling
        :param language: language from ["en", "nl"]
        :return: dictionary with parameter settings as keys and their correction accuracy as values
        """
        # default parameters
        parameters = {'comp_function': 'sum',
                      'include_misspelling': False,
                      'include_oov_candidates': False,
                      'window_size': 6,
                      'reciprocal': False,
                      'remove_stopwords': False,
                      'edit_distance': 1,
                      'oov_penalty': 1.5,
                      'ranking_method': 'context',
                      'k-best': 1}

        dev = Development(parameters, language)

        corrected_list = devcorpus[0]
        detection_list = devcorpus[1]
        detection_contexts = devcorpus[2]

        scores_dict = {}

        start_time = 0
        end_time = 0
        for comp_function in ["sum", "mult", "max"]:
            print("New run")
            run_time = end_time - start_time
            print("Last run took " + str(run_time) + " seconds")
            start_time = time.time()
            dev.comp_function = comp_function
            for include_misspelling in [True, False]:
                dev.include_misspelling = include_misspelling
                for window_size in range(11):
                    dev.window_size = window_size
                    for reciprocal in [True, False]:
                        dev.reciprocal = reciprocal
                        for remove_stopwords in [True, False]:
                            dev.remove_stopwords = remove_stopwords
                            for edit_distance in range(1, 5):
                                dev.edit_distance = edit_distance
                                correction_list = dev.ranking_experiment(detection_list,detection_contexts,
                                                                        candidates_list)
                                accuracy = len([c for i, c in enumerate(correction_list)
                                                if c == corrected_list[i]]) / len(correction_list)
                                parameters = (comp_function, include_misspelling, window_size, reciprocal,
                                              remove_stopwords, edit_distance)
                                scores_dict[parameters] = accuracy

            end_time = time.time()

        return scores_dict

    @staticmethod
    def tune_oov(devcorpus, candidates_list, best_parameters, language):
        """
        Conduct search for best oov penalty for corpus
        :param devcorpus: devcorpus generated with make_devcorpus.py
        :param candidates_list: list of candidate list per misspelling
        :param best_parameters: best parameters for the devcorpus
        :param language: language from ["en", "nl"]
        :return: dictionary with oov penalties as keys and their correction accuracy as values
        """
        dev = Development(best_parameters, language)

        corrected_list = devcorpus[0]
        detection_list = devcorpus[1]
        detection_contexts = devcorpus[2]

        scores_dict = {}

        values = list(range(30))
        values = [value / 10 for value in values]

        for value in values:
            dev.oov_penalty = value
            correction_list = dev.ranking_experiment(detection_list, detection_contexts, candidates_list)
            accuracy = len([c for i, c in enumerate(correction_list)
                            if c == corrected_list[i]]) / len(correction_list)
            scores_dict[value] = accuracy

        return scores_dict

    @staticmethod
    def define_best_parameters(**kwargs):
        """
        Calculates the best parameters or oov penalty averaged over several corpora
        :param kwargs: dictionary with obligatory 'iv' key, with as value a list of scores_dicts calculated with
        Development.grid_search()
        if also 'oov' key, it calculates the optimal oov penalty for all 'iv' and 'oov' scores_dicts calculated with
        Development.tune_oov()
        :return: best parameters or oov penalty averaged over several corpora
        """
        if "oov" not in kwargs.keys():  # grid search
            averaged_scores_dict = {}
            for scores_dict in kwargs['iv']:
                for key in scores_dict:
                    try:
                        averaged_scores_dict[key].append(scores_dict[key])
                    except ValueError:
                        averaged_scores_dict[key] = [scores_dict[key]]

            for key in averaged_scores_dict:
                averaged_scores_dict[key] = mean(averaged_scores_dict[key])

            inverse_dict = {v: k for k, v in averaged_scores_dict.items()}
            best_parameters = inverse_dict[max(inverse_dict.keys())]
            parameters_dict = {}

            parameters_dict["comp_function"] = best_parameters[0]
            parameters_dict["include_misspelling"] = best_parameters[1]
            parameters_dict["window_size"] = best_parameters[2]
            parameters_dict["reciprocal"] = best_parameters[3]
            parameters_dict["remove_stopwords"] = best_parameters[4]
            parameters_dict["edit_distance"] = best_parameters[5]

            return parameters_dict

        else:  # tune oov penalty
            averaged_scores_dict = {}
            for scores_dict in kwargs['iv']:
                for key in scores_dict:
                    try:
                        averaged_scores_dict[key].append(scores_dict[key])
                    except ValueError:
                        averaged_scores_dict[key] = [scores_dict[key]]

            for key in averaged_scores_dict:
                averaged_scores_dict[key] = mean(averaged_scores_dict[key])

            for key in kwargs['oov']:
                averaged_scores_dict[key] = (averaged_scores_dict[key]*9 + kwargs['oov']['key']) / 10  # weighted avg

            inverse_dict = {v: k for k, v in averaged_scores_dict.items()}
            best_parameter = inverse_dict[max(inverse_dict.keys())]

            return best_parameter

    def frequency_analysis(self):
        """
        Calculates the correction accuracies for 3 scenarios: correct replacement has highest frequency (0), second
        highest frequency (1), or lower relative frequency (2) of all candidates
        :return: dictionary with correction accuracy per scenario
        """
        scores_dict = {}

        for j in [0, 1, 2]:
            idxs = []
            for i, candidates in enumerate(self.candidates_list):

                frequencies = [self.frequency_dict[c] if c in self.frequency_dict.keys() else 1 for c in candidates]
                frequencies = np.array(frequencies)
                sorted_candidates = [candidates[i] for i in np.argsort(frequencies)[::-1]]
                if j in [0, 1]:
                    try:
                        eligible_candidates = sorted_candidates[j]
                    except IndexError:
                        eligible_candidates = []
                    if self.corrected_list[i] in eligible_candidates:
                        idxs.append(i)
                else:
                    try:
                        eligible_candidates = sorted_candidates[2:]
                    except IndexError:
                        eligible_candidates = []
                    if self.corrected_list[i] in eligible_candidates:
                        idxs.append(i)

            correction_list = [x for i, x in enumerate(self.correction_list) if i in idxs]
            corrected_list = [x for i, x in enumerate(self.corrected_list) if i in idxs]

            accuracy = len([c for i, c in enumerate(correction_list) if c == corrected_list[i]]) / len(correction_list)
            scores_dict[j] = accuracy

            print('Top k')
            print(j)
            print('Amount of instances')
            print(len(idxs))
            print('Accuracy')
            print(accuracy)

        print(scores_dict)

        return scores_dict
