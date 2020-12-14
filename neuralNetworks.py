# %%
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.patches as mpatches
import time

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import collections

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report, confusion_matrix, plot_confusion_matrix
from collections import Counter
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, cross_val_score, GridSearchCV, cross_val_predict
from sklearn.preprocessing import RobustScaler
from scipy.stats import norm

import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.callbacks import ModelCheckpoint

import itertools
# %%
dataset = pd.read_csv('./dataset/creditcard.csv')
dataset.head()
# %%
dataset.describe()
# %%
robustScaler = RobustScaler()
dataset['scaled_amount'] = robustScaler.fit_transform(
    dataset['Amount'].values.reshape(-1, 1))
dataset['scaled_time'] = robustScaler.fit_transform(
    dataset['Time'].values.reshape(-1, 1))
# %%
dataset.drop(['Amount', 'Time'], axis=1, inplace=True)
dataset.head()
# %%
X = dataset.drop(['Class'], axis=1)
Y = dataset['Class']
# %%
SKfold = StratifiedKFold(random_state=42)
for train_index, test_index in SKfold.split(X, Y):
    og_X_train, og_X_test = X.iloc[train_index], X.iloc[test_index]
    og_Y_train, og_Y_test = Y.iloc[train_index], Y.iloc[test_index]

# %%
og_X_train = og_X_train.values
og_X_test = og_X_test.values
og_Y_train = og_Y_train.values
og_Y_test = og_Y_test.values
# %%
dataset = dataset.sample(frac=1, random_state=42)
fraud = dataset.loc[dataset['Class'] == 1]
normal = dataset.loc[dataset['Class'] == 0][:492]
nd_dataset = pd.concat([fraud, normal])
nd_dataset = nd_dataset.sample(frac=1, random_state=42)
nd_dataset.head()
# %%
nd_X = nd_dataset.drop("Class", axis=1)
nd_Y = nd_dataset["Class"]

# %%
nd_Xtrain, nd_Xtest, nd_Ytrain, nd_Ytest = train_test_split(
    nd_X, nd_Y, random_state=42, test_size=0.2)
nd_Xtrain = nd_Xtrain.values
nd_Xtest = nd_Xtest.values
nd_Ytrain = nd_Ytrain.values
nd_Ytest = nd_Ytest.values

# %%
n_inputs = nd_Xtrain.shape[1]
undersample_model = Sequential([
    Dense(n_inputs, input_shape=(n_inputs,), activation="relu"),
    Dense(32, activation="relu"),
    Dense(2, activation="softmax")
])
# %%
undersample_model.summary()
# %%
undersample_model.compile(
    Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=["accuracy"])
modelcheckpoint = ModelCheckpoint(
    "models/undersample_model.h5", save_best_only=True, monitor="val_acc")
undersample_model.fit(nd_Xtrain, nd_Ytrain, validation_split=0.2, epochs=20,
                      batch_size=25, shuffle=True, verbose=2, callbacks=[modelcheckpoint])

# %%
undersample_pred = undersample_model.predict(og_X_test, verbose=2)
# %%
undersample_pred_classes = undersample_model.predict_classes(
    og_X_test, verbose=2)
# %%
confmat = confusion_matrix(og_Y_test, undersample_pred_classes)
print(confmat)
# %%


def plotTensorflowConfmat(confmat, classes):
    plt.imshow(confmat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    for i, j in itertools.product(range(confmat.shape[0]), range(confmat.shape[1])):
        plt.text(j, i, format(confmat[i, j], '.2f'),
                 horizontalalignment='center', color='black')


# %%
classes = ["Normal", "Fraud"]
plotTensorflowConfmat(confmat, classes)

# %%
sm = SMOTE(sampling_strategy="minority", random_state=42)
sm_X_train, sm_Y_train = sm.fit_sample(og_X_train, og_Y_train)
# %%
sm_X_train.shape
# %%
n_inputs = sm_X_train.shape[1]
smote_model = Sequential([
    Dense(n_inputs, input_shape=(n_inputs,), activation='relu'),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
])
# %%
smote_model.summary()
# %%
smote_model.compile(
    Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
modelcheckpoint = ModelCheckpoint(
    'models/smote_model.h5', save_best_only=True, monitor='val_acc')
smote_model.fit(sm_X_train, sm_Y_train, validation_split=0.2, batch_size=25,
                epochs=20, verbose=2, shuffle=True, callbacks=[modelcheckpoint])
# %%
smote_model.save('models/smote_model.h5')
# %%
smote_pred_classes = smote_model.predict_classes(og_X_test)
# %%
confmat = confusion_matrix(og_Y_test, smote_pred_classes)
print(confmat)
# %%
plotTensorflowConfmat(confmat, classes)
