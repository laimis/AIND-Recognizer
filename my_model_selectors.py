import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN

    I struggled with discoverint p formula, but a good lady on slack informed me with this:
    # There is one thing a little different for our project though... 
    # p = n*(n-1) + (n-1) + 2*d*n
    #         = n^2 + 2*d*n - 1
    #
    # https://ai-nd.slack.com/archives/C3TSZ56U8/p1491489096694280
    # and this
    # https://ai-nd.slack.com/archives/C3V8A1MM4/p1493912001029860

    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        selectedModel = None
        smallestScore = 1000000

        for component_count in range(self.min_n_components, self.max_n_components + 1):
    
            bic = None

            try:
                hmm = GaussianHMM(n_components=component_count, n_iter=1000)
                
                hmm.fit(self.X, self.lengths)
                
                logL = hmm.score(self.X, self.lengths)
                logN = np.log(sum(self.lengths))

                # see the comments above how I got the p formula                
                p = component_count ** 2 + 2 * len(self.sequences) * component_count  - 1

                bic = -2 * logL + p * logN

            except:
                # print("exception",component_count)
                pass
            
            if bic == None:
                continue
                
            if bic < smallestScore:

                selectedModel = hmm
                smallestScore = bic 
        
        return selectedModel


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))

    To understand DIC, I referred to these discussions:
    https://ai-nd.slack.com/archives/C3V8A1MM4/p1491870173200006
    https://ai-nd.slack.com/archives/C3V8A1MM4/p1494618695642425
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        selectedModel = None
        bestScore = -1000000

        for component_count in range(self.min_n_components, self.max_n_components + 1):
    
            dic = None

            try:
                hmm = GaussianHMM(n_components=component_count, n_iter=1000)
                
                hmm.fit(self.X, self.lengths)
                
                score = hmm.score(self.X, self.lengths)
                
                mean = 0.0
                for word, Xlengths in self.hwords.items():
                    if word != self.this_word:
                        mean += hmm.score(Xlengths[0], Xlengths[1])
                
                mean = mean/(len(self.hwords)-1)

                dic = score - mean

            except:
                # print("exception",component_count)
                pass
            
            if dic == None:
                continue
                
            if dic > bestScore:

                selectedModel = hmm
                bestScore = dic 
        
        return selectedModel


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    more info:
    http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # try out a range of components
        # for each component count, create hmm, train using kfold data split approach
        # record the best

        selectedModel = None
        bestScore = -1000000

        for component_count in range(self.min_n_components, self.max_n_components + 1):

            numberOfSplits = 3
            if component_count < numberOfSplits:
                numberOfSplits = component_count

            if len(self.sequences) < component_count:
                numberOfSplits = len(self.sequences)

            if numberOfSplits == 1:
                continue

            split_method = KFold(n_splits = numberOfSplits)

            scores = []

            for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                
                try:
                    hmm = GaussianHMM(n_components=component_count, n_iter=1000)

                    # training data
                    train, lengths = combine_sequences(cv_train_idx, self.sequences) 
                    model = hmm.fit(train, lengths)

                    # test data for score
                    test, lengths = combine_sequences(cv_test_idx, self.sequences)
                    score = hmm.score(test, lengths)
                    
                    scores.append(score)
                except:
                    # print("exception",component_count)
                    pass
            
            # sometimes there is just no data
            if len(scores) == 0:
                continue

            avgScore = float(sum(scores)) / len(scores)

            if avgScore > bestScore:

                selectedModel = hmm
                bestScore = avgScore 
        
        return selectedModel