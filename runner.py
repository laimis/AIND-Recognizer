import numpy as np
import pandas as pd
from asl_data import AslDb


asl = AslDb() # initializes the database

asl.df.ix[98,1]  # look at the data available for an individual frame

asl.df['grnd-ry'] = asl.df['right-y'] - asl.df['nose-y']

from asl_utils import test_features_tryit
# TODO add df columns for 'grnd-rx', 'grnd-ly', 'grnd-lx' representing differences between hand and nose locations

# grnd-rx
asl.df['grnd-rx'] = asl.df['right-x'] - asl.df['nose-x']

# grnd-ly
asl.df['grnd-ly'] = asl.df['left-y'] - asl.df['nose-y']

# grnd-lx
asl.df['grnd-lx'] = asl.df['left-x'] - asl.df['nose-x']

# test the code
# test_features_tryit(asl)


# collect the features into a list
features_ground = ['grnd-rx','grnd-ry','grnd-lx','grnd-ly']
 #show a single set of features for a given (video, frame) tuple
[asl.df.ix[98,1][v] for v in features_ground]

training = asl.build_training(features_ground)
print("Training words: {}".format(training.words))

df_means = asl.df.groupby('speaker').mean()

asl.df['left-x-mean']= asl.df['speaker'].map(df_means['left-x'])

from asl_utils import test_std_tryit
# TODO Create a dataframe named `df_std` with standard deviations grouped by speaker
df_std = asl.df.groupby('speaker').std()

# test the code
# test_std_tryit(df_std)

# TODO add features for normalized by speaker values of left, right, x, y
# Name these 'norm-rx', 'norm-ry', 'norm-lx', and 'norm-ly'
# using Z-score scaling (X-Xmean)/Xstd

# https://en.wikipedia.org/wiki/Standard_score
#	normalized = (x - mean) / standard deviation
#	so we need all means for rx, lx, ry, ly by speaker? and then deviations for all

features_norm = ['norm-rx', 'norm-ry', 'norm-lx','norm-ly']

# copied from the above and then covering the rest
asl.df['right-x-mean']= asl.df['speaker'].map(df_means['right-x'])
asl.df['right-y-mean']= asl.df['speaker'].map(df_means['right-y'])
asl.df['left-x-mean']= asl.df['speaker'].map(df_means['left-x'])
asl.df['left-y-mean']= asl.df['speaker'].map(df_means['left-y'])

# then we need standard deviations using df_std from the above
asl.df['right-x-std']= asl.df['speaker'].map(df_std['right-x'])
asl.df['right-y-std']= asl.df['speaker'].map(df_std['right-y'])
asl.df['left-x-std']= asl.df['speaker'].map(df_std['left-x'])
asl.df['left-y-std']= asl.df['speaker'].map(df_std['left-y'])

# now plug into formula
asl.df['norm-rx'] = (asl.df['right-x'] - asl.df['right-x-mean'])/asl.df['right-x-std']
asl.df['norm-ry'] = (asl.df['right-y'] - asl.df['right-y-mean'])/asl.df['right-y-std']
asl.df['norm-lx'] = (asl.df['left-x'] - asl.df['left-x-mean'])/asl.df['left-x-std']
asl.df['norm-ly'] = (asl.df['left-y'] - asl.df['left-y-mean'])/asl.df['left-y-std']



# TODO add features for polar coordinate values where the nose is the origin
# Name these 'polar-rr', 'polar-rtheta', 'polar-lr', and 'polar-ltheta'
# Note that 'polar-rr' and 'polar-rtheta' refer to the radius and angle

# forgot what polar coordinates are, refresher: http://mathworld.wolfram.com/PolarCoordinates.html
#  r - radius, theta - angle
# r = sqrt(x^2 + y^2)
# theta = tan^(-1)(y/x).

features_polar = ['polar-rr', 'polar-rtheta', 'polar-lr', 'polar-ltheta']

asl.df['polar-rr'] = (asl.df['grnd-rx']**2 + asl.df['grnd-ry']**2) ** (1/2)
asl.df['polar-lr'] = (asl.df['grnd-lx']**2 + asl.df['grnd-ly']**2) ** (1/2)

# arctan2 = https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.arctan2.html
asl.df['polar-rtheta'] = np.arctan2(asl.df['grnd-rx'],asl.df['grnd-ry'])
asl.df['polar-ltheta'] = np.arctan2(asl.df['grnd-lx'],asl.df['grnd-ly'])


features_delta = ['delta-rx', 'delta-ry', 'delta-lx', 'delta-ly']

# diff function: https://docs.scipy.org/doc/numpy-1.10.4/reference/generated/numpy.diff.html
# cool!
# add fillna at the end so that we don't ddeal with na values
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.fillna.html
# is zero right?
asl.df['delta-rx'] = asl.df['right-x'].diff().fillna(0)
asl.df['delta-ry'] = asl.df['right-y'].diff().fillna(0)
asl.df['delta-lx'] = asl.df['left-x'].diff().fillna(0)
asl.df['delta-ly'] = asl.df['left-y'].diff().fillna(0)


features_custom = ['rescale-rx', 'rescale-ry', 'rescale-lx','rescale-ly']

df_min = asl.df.groupby('speaker').min()
df_max = asl.df.groupby('speaker').max()

asl.df['right-x-min']= asl.df['speaker'].map(df_min['right-x'])
asl.df['right-y-min']= asl.df['speaker'].map(df_min['right-y'])
asl.df['left-x-min']= asl.df['speaker'].map(df_min['left-x'])
asl.df['left-y-min']= asl.df['speaker'].map(df_min['left-y'])

asl.df['right-x-max']= asl.df['speaker'].map(df_max['right-x'])
asl.df['right-y-max']= asl.df['speaker'].map(df_max['right-y'])
asl.df['left-x-max']= asl.df['speaker'].map(df_max['left-x'])
asl.df['left-y-max']= asl.df['speaker'].map(df_max['left-y'])

# now plug into formula
asl.df['rescale-rx'] = (asl.df['right-x'] - asl.df['right-x-min']) / (asl.df['right-x-max'] - asl.df['right-x-min'])
asl.df['rescale-ry'] = (asl.df['right-y'] - asl.df['right-y-min']) / (asl.df['right-y-max'] - asl.df['right-x-min'])
asl.df['rescale-lx'] = (asl.df['left-x'] - asl.df['left-x-min']) / (asl.df['left-x-max'] - asl.df['right-x-min']) 
asl.df['rescale-ly'] = (asl.df['left-y'] - asl.df['left-y-min']) / (asl.df['left-y-max'] - asl.df['right-x-min']) 

import warnings
from hmmlearn.hmm import GaussianHMM

def train_a_word(word, num_hidden_states, features):
    
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    training = asl.build_training(features)  
    X, lengths = training.get_word_Xlengths(word)
    model = GaussianHMM(n_components=num_hidden_states, n_iter=1000).fit(X, lengths)
    logL = model.score(X, lengths)
    return model, logL

# demoword = 'BOOK'
# model, logL = train_a_word(demoword, 3, features_ground)
# print("Number of states trained in model for {} is {}".format(demoword, model.n_components))
# print("logL = {}".format(logL))


def show_model_stats(word, model):
    print("Number of states trained in model for {} is {}".format(word, model.n_components))    
    variance=np.array([np.diag(model.covars_[i]) for i in range(model.n_components)])    
    for i in range(model.n_components):  # for each hidden state
        print("hidden state #{}".format(i))
        print("mean = ", model.means_[i])
        print("variance = ", variance[i])
        print()
    
# show_model_stats(demoword, model)

from my_model_selectors import SelectorConstant

# training = asl.build_training(features_ground)  # Experiment here with different feature sets defined in part 1
# word = 'VEGETABLE' # Experiment here with different words
# model = SelectorConstant(training.get_all_sequences(), training.get_all_Xlengths(), word, n_constant=3).select()
# print("Number of states trained in model for {} is {}".format(word, model.n_components))

from sklearn.model_selection import KFold

# training = asl.build_training(features_ground) # Experiment here with different feature sets
# word = 'VEGETABLE' # Experiment here with different words
# word_sequences = training.get_word_sequences(word)
# split_method = KFold()
# for cv_train_idx, cv_test_idx in split_method.split(word_sequences):
#     print("Train fold indices:{} Test fold indices:{}".format(cv_train_idx, cv_test_idx))  # view indices of the folds

words_to_train = ['FISH', 'BOOK', 'VEGETABLE', 'FUTURE', 'JOHN']
import timeit


def train_with_cv():

	from my_model_selectors import SelectorCV

	training = asl.build_training(features_ground)  # Experiment here with different feature sets defined in part 1
	sequences = training.get_all_sequences()
	Xlengths = training.get_all_Xlengths()
	for word in words_to_train:
		start = timeit.default_timer()
		model = SelectorCV(sequences, Xlengths, word, 
						min_n_components=2, max_n_components=15, random_state = 14).select()
		end = timeit.default_timer()-start
		if model is not None:
			print("Training complete for {} with {} states with time {} seconds".format(word, model.n_components, end))
		else:
			print("Training failed for {}".format(word))

def train_with_bic():

	print("training with bic")

	from my_model_selectors import SelectorBIC

	training = asl.build_training(features_ground)  # Experiment here with different feature sets defined in part 1
	sequences = training.get_all_sequences()
	Xlengths = training.get_all_Xlengths()
	for word in words_to_train:
		start = timeit.default_timer()
		model = SelectorBIC(sequences, Xlengths, word, 
						min_n_components=2, max_n_components=15, random_state = 14).select()
		end = timeit.default_timer()-start
		if model is not None:
			print("Training complete for {} with {} states with time {} seconds".format(word, model.n_components, end))
		else:
			print("Training failed for {}".format(word))

def train_with_dic():
	print ("training with dic")

	from my_model_selectors import SelectorDIC

	training = asl.build_training(features_ground)  # Experiment here with different feature sets defined in part 1
	sequences = training.get_all_sequences()
	Xlengths = training.get_all_Xlengths()
	for word in words_to_train:
		start = timeit.default_timer()
		model = SelectorDIC(sequences, Xlengths, word, 
						min_n_components=2, max_n_components=15, random_state = 14).select()
		end = timeit.default_timer()-start
		if model is not None:
			print("Training complete for {} with {} states with time {} seconds".format(word, model.n_components, end))
		else:
			print("Training failed for {}".format(word))


train_with_dic()

