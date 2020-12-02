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
