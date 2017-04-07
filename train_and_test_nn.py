# Initial Code referenced from http://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.callbacks import ModelCheckpoint
from keras.wrappers.scikit_learn import KerasRegressor
from keras.optimizers import SGD, RMSprop
from keras.models import load_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import time
from collections import Counter
import csv


'''
LOADS AND TRAINS A NEURAL NETWORK; PREDICTS CLASSIFICATION LABELS FOR TEST DATA
'''


# Load training and test data tables
TRAINING = pd.read_csv('master_training_final_2.csv', sep=',',encoding='utf-8',header=None)
TEST = pd.read_csv('kaggle_test_final.csv', sep=',',encoding='utf-8',header=None)


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

def results_to_csv(filename, results):
    '''Function to write prediction results to csv '''
    with open(filename, 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(["Id", "Category"])
        index = 0
        for val in results:
            writer.writerow([index, int(val)])
            index +=1

def create_neural_network(training, test):
''' Creates, trains a neural network, runs predictions '''

    # split data into input (X) and output (Y) variables
    dataset = training.values
    input_dimension = len(dataset[0])-1
    X = training.iloc[:,:input_dimension].values
    Y = training.iloc[:,input_dimension:].values

    # Prepares test data for predictions
    TEST_DATA = test.values
    TEST_X = test.iloc[:,0:]#.astype(np.int32)



	# define base mode
	# Guide to Sequential Model: https://keras.io/getting-started/sequential-model-guide/

	# Sequential model constructer takes a list of layer instances
	#   these layers can be specified ina direct constructor OR sequentially with an "add" method
	#   the instance layer should specifiy the dimensions of the input shape add(Dense(32, input_shape=(784,)))
	#

    def baseline_model():

        # create model
        model = Sequential()

        # Add base layer, hidden layers and ouput layers
        model.add(Dense(input_dimension, input_dim=input_dimension, activation='relu'))
        model.add(Dense(500, init='normal', activation='relu'))
        model.add(Dense(75, init='normal', activation='relu'))
        model.add(Dense(1, init='normal',activation='sigmoid'))

        # Option for loading prexisting nn weights
        #model.load_weights("model_v2.h5")

        # Compile nn for binary classification
        rms = RMSprop()
        model.compile(loss='binary_crossentropy', optimizer=rms, metrics=['accuracy'])


        # Train model
        history = model.fit(X, Y, nb_epoch=100, batch_size=50)

        # Save for model weights for later use
        model.save_weights("model_v6.h5")

        # Use model to predict classificaiton on test data
        predictions = model.predict(np.array(TEST_X))

        # round predictions
        final_predictions = []
        for x in predictions:
            if x[0] > 0.5:
                final_predictions.append(1)
            else:
                final_predictions.append(0)

        # Format and print prediction results to csv
        result_counter = Counter()
        for x in final_predictions:
            result_counter[x] +=1
        results_to_csv('kaggle_test_results_4.csv', final_predictions)

        return model

    model = baseline_model()

	# Optional: evaluate model with standardized dataset using K-folds Validation
    '''estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size=5, verbose=1)
    #kfold = KFold(n_splits=10, random_state=seed)
    #results = cross_val_score(estimator, X, Y, cv=kfold)
    #print results
    #print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
    #print time.clock()'''


if __name__ == "__main__":

    try:
        create_neural_network(TRAINING[:40000], TEST)
    except:
        raise
