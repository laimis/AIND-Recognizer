import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    # need init
    allXLengths = test_set.get_all_Xlengths()
    for n in range(len(allXLengths)):
        probabilities.append(dict())
        guesses.append(None)
    
    for index, value in allXLengths.items():
        
        bestScore = -1000000
        bestCandidate = None

        for word, model in models.items():
            try:
                score = model.score(value[0], value[1])
                
                probabilities[index][word] = score

                if score > bestScore:
                    bestScore = score
                    bestCandidate = word

            except Exception as e:
                probabilities[index][word] = -100000

        guesses[index] = bestCandidate

    return probabilities, guesses