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
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from collections import Counter
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import RobustScaler

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
unique_train, train_count = np.unique(og_Y_train, return_counts=True)
unique_test, test_count = np.unique(og_Y_test, return_counts=True)

print(train_count/len(og_Y_train))
print(test_count/len(og_Y_test))
# %%
dataset = dataset.sample(frac=1, random_state=42)
fraud = dataset.loc[dataset['Class'] == 1]
normal = dataset.loc[dataset['Class'] == 0][:492]
nd_dataset = pd.concat([fraud, normal])
nd_dataset = nd_dataset.sample(frac=1, random_state=42)
nd_dataset.head()
# %%
sns.countplot('Class', data=nd_dataset)
plt.title("Distribution of classes")
plt.show()
# %%
corr = nd_dataset.corr()
sns.heatmap(corr, cmap='coolwarm_r', annot_kws={'size': 20})
plt.show()

# %%
positive_corr = ["V2", "V4", "V11", "V19"]
negative_corr = ["V10", "V12", "V14", "V16"]
# %%
f, axes = plt.subplots(ncols=4, figsize=(20, 4))
f.suptitle("Positive Correlation")
for i, feature in enumerate(positive_corr):
    sns.boxplot(x="Class", y=feature, data=nd_dataset, ax=axes[i])
    axes[i].set_title("Class VS " + feature)

# %%
f, axes = plt.subplots(ncols=4, figsize=(20, 4))
f.suptitle("Negative Correlation")
for i, feature in enumerate(negative_corr):
    sns.boxplot(x="Class", y=feature, data=nd_dataset, ax=axes[i])
    axes[i].set_title("Class VS " + feature)
# %%
