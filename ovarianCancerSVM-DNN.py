import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import pandas as pd
from pandas import Series
import numpy as np
import scipy
from scipy.fft import fft, fftfreq
from nfft import nfft
from pynufft import NUFFT
from pyopenms import *
import pywt
from sklearn.metrics.pairwise import euclidean_distances
from scipy import signal
import scaleogram as scg
from sklearn.decomposition import PCA

from scipy.misc import electrocardiogram
from scipy.signal import find_peaks
import sys
sys.path.append(".")
from peak_finder import PeakFinder
from matplotlib import pyplot

from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import datasets, linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import StratifiedKFold 
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
##Keras

from keras.layers import Dense, Dropout
from keras.models import Model, Sequential
from keras.wrappers.scikit_learn import KerasRegressor
import ast
##


def data_handle():
    cancer_entries = os.listdir('C:/Users/arman/Workspace/CS234/Project/OvarianCD_PostQAQC/Cancer')                                     # Directory containing Cancerous Mass Spec Files
    normal_entries = os.listdir('C:/Users/arman/Workspace/CS234/Project/OvarianCD_PostQAQC/Normal')                                     # Directory containing Normal Mass Spec Files
    raw = []
    classes = []
    discard_counter = 0
    c_counter = 0
    n_counter = 0
    bins =[]
    for entry in cancer_entries:
        df = pd.read_csv('C:/Users/arman/Workspace/CS234/Project/OvarianCD_PostQAQC/Cancer/' + entry, sep = "\t", header = None)
        x = df[1]
        x = x.values.flatten()      # Convert to Numpy 1d Array

        # Finding the peaks based on threshold difference
        peak_finder = PeakFinder(x)
        peak_finder.filter_by_threshold(tmin=12)
        peak_indexes = peak_finder.peaks                    # Indexes

        #### Plotting Peak Finder ####    
        # peak_finder.plot()
        # plt.show()
        #### Plotting Peak Finder ####
        
        num_peaks = 80
        rows = df.iloc[peak_indexes, :]                     # Retrive mass based on peak index

        rows = rows.nlargest(num_peaks,1)                  # Retrieving Min of peaks 
        rows = rows.values.tolist()
        if len(rows)>= num_peaks:
            raw.append(rows)
            classes.append(1)
            c_counter+=1
        else:
            discard_counter+=1

    for entry in normal_entries:
        df = pd.read_csv('C:/Users/arman/Workspace/CS234/Project/OvarianCD_PostQAQC/Normal/' + entry, sep = "\t", header = None)
        #print(df)
        x = df[1]
        x = x.values.flatten()      # Convert to Numpy 1d Array

        # Finding the peaks based on threshold difference
        peak_finder = PeakFinder(x)
        peak_finder.filter_by_threshold(tmin=7)
        peak_indexes = peak_finder.peaks                    # Indexes
        #print(peak_indexes)
        #### Plotting Peak Finder ####    
        # peak_finder.plot()
        # plt.show()
        ####                      ####
        
        num_peaks = 80
        rows = df.iloc[peak_indexes, :]                     # Retrive mass based on peak index
        #print(type(rows))

        rows = rows.nsmallest(num_peaks,1)
        rows = rows.values.tolist()
        if len(rows)>= num_peaks:
            raw.append(rows)
            classes.append(0)
            n_counter += 1
        else:
            discard_counter+=1
    print("ccount: ", c_counter)
    print("ncount: ", n_counter)
    print(discard_counter, " Mass Spec samples discarded")
    
    ## Distribution Chart 
    bins.append(c_counter)
    bins.append(n_counter)
    x=np.arange(2)
    plt.bar(x,height=bins)
    plt.xticks(x, ['Cancer','Normal'])
    plt.show()
    
    
    results = pd.DataFrame(raw)             # Raw Features
    print(results)
    classes = pd.DataFrame(classes)         #Target aka cancer = 1/normal = 0
    remove_n = c_counter - n_counter

    ## Randomly Balancing Datasets <-- Perfomance not great/fluctuates greatly. Use weights in SVM    
    # if remove_n > 0:
    #     print("results_old ", len(results))
    #     print("results_old ", len(classes))
    #     #Randomly remove cancer data to balance dataset
    #     drop_indices = np.random.choice(results.index[:c_counter], remove_n, replace=False)
    #     print("dropped: ",drop_indices)
    #     test_remove = classes.iloc[drop_indices, :]
    #     results = results.drop(drop_indices)
    #     classes = classes.drop(drop_indices)
           
    #     print(test_remove)
    #     print("results_new ", len(results))
    #     print("results_new ", len(classes))

    #     plt.bar(x,height=[n_counter,c_counter-remove_n])
    #     plt.xticks(x, ['Cancer','Normal'])
    #     plt.show()
    results.to_csv("C:/Users/arman/Workspace/CS234/Project/raw_data_combined.csv",index=False)
    classes.to_csv("C:/Users/arman/Workspace/CS234/Project/classes.csv",index = False)
    return results, classes

def get_test_data():
    raw = []
    df = pd.read_csv('C:/Users/arman/Workspace/CS234/Project/OvarianCD_PostQAQC/Normal/daf-0186.txt', sep = "\t", header = None,dtype='float64')
    x = df[1]
    x = x.values.flatten()      # Convert to Numpy 1d Array

    # Finding the peaks based on threshold difference
    peak_finder = PeakFinder(x)
    peak_finder.filter_by_threshold(tmin=12)
    peak_indexes = peak_finder.peaks                    # Indexes
    #print(peak_indexes)
    #### Plotting Peak Finder ####    
    #peak_finder.plot()
    #plt.show()
    ####                      ####
    
    num_peaks = 40
    rows = df.iloc[peak_indexes, :]                     # Retrive mass based on peak index
    #print(type(rows))

    rows = rows.nlargest(num_peaks,1)
    rows = rows.values.tolist()
    raw.append(rows)
    results = pd.DataFrame(raw)
    pca = PCA(n_components = 1) 
    results = results.applymap(lambda x: pca.fit_transform(np.array(x).reshape((-1,1)))[0][0])
    return results

def get_models():
	models = dict()
	for i in range(1,80):
		steps = [('svd', TruncatedSVD(n_components=i)), ('m', LogisticRegression())]
		models[str(i)] = Pipeline(steps=steps)
	return models

def evaluate_model(model, X, y):
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=0)
	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
	return scores

def deep_model(optimizer='adagrad',
                 kernel_initializer='normal', 
                 dropout=0.2):
    model = Sequential()
    model.add(Dense(64,activation='relu',kernel_initializer=kernel_initializer))
    model.add(Dropout(dropout))
    model.add(Dense(1,activation='sigmoid',kernel_initializer=kernel_initializer))

    model.compile(loss='binary_crossentropy',optimizer=optimizer, metrics=['accuracy'])

    return model

if __name__ == '__main__':
    make_data = True
    if make_data:
        x,y = data_handle()
    else:
        x = pd.read_csv("C:/Users/arman/Workspace/CS234/Project/raw_data_combined.csv", quotechar='"', sep=',',converters={1:ast.literal_eval})
        y = pd.read_csv("C:/Users/arman/Workspace/CS234/Project/classes.csv",dtype='float64')
    # x = x.values.replace("'", '').apply(ast.literal_eval)         < -- basic logic for fixing csv formatting errors

    pca = PCA(n_components = 1)

    ## SCALING -- Does not seem to affect performance.. ## 
    # scaler = StandardScaler()
    # pt = PowerTransformer()
    # qt = QuantileTransformer(output_distribution='normal')
    #x = x.applymap(lambda x: qt.fit_transform(np.array(x).reshape((-1,1))))
    #pca = KernelPCA(n_components = 1, kernel ='poly')
    
        
    x = x.applymap(lambda x: pca.fit_transform(np.array(x).reshape((-1,1)))[0][0])                  # Reshape array and perform PCA to reduce dimensionality
    x = x.to_numpy()
    y = y.to_numpy()
    y = y.reshape((-1,))        # Reshaping to ndarray with shape (len(y),)

    x,y = shuffle(x,y)          # Shuffle to mix up dataset

    ## Single Test Prediction ##
            # test = get_test_data()
            # print(test)
            # clf = svm.SVC()
            # clf.fit(x, y)
            # res = clf.predict(test)
            # print(res)
    ## Single Test Prediction ##

    
    ## START SVD with Stratified KFold Cross Validation ##
    models = get_models()

    results, names = list(), list()
    for name, model in models.items():
        scores = evaluate_model(model, x, y)
        results.append(scores)
        names.append(name)
        print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))

    pyplot.boxplot(results, labels=names, showmeans=True)
    pyplot.xticks(rotation=45)
    pyplot.show()
    ## END SVD with Stratified KFold Cross Validation ##

    ## SVD LOGIC ##

    ##pipeline##
    # steps = [('svd', TruncatedSVD(n_components=15)), ('m', LogisticRegression())]
    # model = Pipeline(steps=steps)
    ##pipeline##
    model = TruncatedSVD(n_components=17, n_iter=100 ,random_state=0)
    model_svd = model.fit_transform(x,y)

    
    ## SVD LOGIC ##

    ## START SVM LOGIC ##
    clf = svm.SVC(kernel='sigmoid',verbose=1)  
    # Cs = np.logspace(-6, -1, 10)
    # clf1 = GridSearchCV(estimator=clf, param_grid=dict(C=Cs), n_jobs=-1)
    # clf1 = clf1.best_estimator_

    X_train, X_test, y_train, y_test = train_test_split(model_svd, y, test_size=0.3,random_state=0)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    # print(y_pred)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    print("Precision:",metrics.precision_score(y_test, y_pred))
    print("Recall:",metrics.recall_score(y_test, y_pred))
    ## END SVM LOGIC ##

    ## DEEP NEURAL NET HYPERPARAMETERS ##

    # clf = KerasRegressor(build_fn=deep_model)

    # scaler = StandardScaler()

    # # parameter grid
    # param_grid = {
    #     'clf__optimizer':['rmsprop','adam','adagrad'],
    #     'clf__epochs':[4,8,12,16],
    #     'clf__dropout':[0.1,0.2,0.3],
    #     'clf__kernel_initializer':['glorot_uniform','normal','uniform']
    # }

    # pipeline = Pipeline([
    #     ('preprocess',scaler),
    #     ('clf',clf)
    # ])

    # grid = GridSearchCV(pipeline, cv=3, param_grid=param_grid)
    # grid.fit(X_train, y_train)

    # print("Best: %f using %s" % (grid.best_score_, grid.best_params_))
    # means = grid.cv_results_['mean_test_score']
    # stds = grid.cv_results_['std_test_score']
    # params = grid.cv_results_['params']
    # for mean, stdev, param in zip(means, stds, params):
    #     print("%f (%f) with: %r" % (mean, stdev, param))

    ## DEEP NEURAL NET HYPERPARAMETERS ##

    ## Deep NEURAL NET ACTUAL ##
    model = deep_model()
    history = model.fit(X_train, y_train,validation_data=(X_test,y_test), epochs=100, batch_size=1)
    score,accuracy = model.evaluate(X_test, y_test)
    print('Accuracy: %.2f' % (accuracy*100))
    pyplot.title('Accuracy')
    pyplot.plot(history.history['accuracy'], label='train')
    pyplot.plot(history.history['val_accuracy'], label='test')
    pyplot.legend()
    pyplot.show()
    # predict probabilities for test set
    yhat_probs = model.predict(X_test, verbose=0)
    # predict classes for test set
    yhat_classes = model.predict_classes(X_test, verbose=0)

    yhat_probs = yhat_probs[:, 0]
    yhat_classes = yhat_classes[:, 0]

    accuracy = accuracy_score(y_test, yhat_classes)
    print('Accuracy: %f' % accuracy)

    precision = precision_score(y_test, yhat_classes)
    print('Precision: %f' % precision)

    recall = recall_score(y_test, yhat_classes)
    print('Recall: %f' % recall)
