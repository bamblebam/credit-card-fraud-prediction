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
from scipy.stats import norm

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
# positive corr
f, axes = plt.subplots(ncols=4, figsize=(20, 4))
f.suptitle("Positive Correlation")
for i, feature in enumerate(positive_corr):
    sns.boxplot(x="Class", y=feature, data=nd_dataset, ax=axes[i])
    axes[i].set_title("Class VS " + feature)

# %%
# negative corr
f, axes = plt.subplots(ncols=4, figsize=(20, 4))
f.suptitle("Negative Correlation")
for i, feature in enumerate(negative_corr):
    sns.boxplot(x="Class", y=feature, data=nd_dataset, ax=axes[i])
    axes[i].set_title("Class VS " + feature)
# %%
corr['Class'].sort_values()
# %%
# positive norm dist
f, axes = plt.subplots(ncols=4, figsize=(20, 4))
f.suptitle("Normal Distribution")
for i, feature in enumerate(positive_corr):
    fraud_dist = nd_dataset[feature].loc[nd_dataset["Class"] == 1].values
    sns.distplot(fraud_dist, ax=axes[i], fit=norm)
    axes[i].set_title(feature+" Distribution")
# %%
# negative norm dist
f, axes = plt.subplots(ncols=4, figsize=(20, 4))
f.suptitle("Normal Distribution")
for i, feature in enumerate(negative_corr):
    fraud_dist = nd_dataset[feature].loc[nd_dataset["Class"] == 1].values
    sns.distplot(fraud_dist, ax=axes[i], fit=norm)
    axes[i].set_title(feature+" Distribution")
# %%
# positive outlier removal
for i, feature in enumerate(positive_corr):
    fraud_dist = nd_dataset[feature].loc[nd_dataset["Class"] == 1].values
    q25, q75 = np.percentile(fraud_dist, 25), np.percentile(fraud_dist, 75)
    iqr = q75-q25
    cutoff = iqr*1.5
    upper_limit = q75+cutoff
    lower_limit = q25-cutoff
    outlier_list = [x for x in fraud_dist if x <
                    lower_limit or x > upper_limit]
    nd_dataset = nd_dataset.drop(nd_dataset[(nd_dataset[feature] > upper_limit) | (
        nd_dataset[feature] < lower_limit)].index)
    print(f"Lower limit {lower_limit}")
    print(f"Upper limit {upper_limit}")
    print(outlier_list)
    print(f"Outliers removed {len(outlier_list)}")
    print("\n\n")

# %%
# negative outlier removal
for i, feature in enumerate(negative_corr):
    fraud_dist = nd_dataset[feature].loc[nd_dataset["Class"] == 1].values
    q25, q75 = np.percentile(fraud_dist, 25), np.percentile(fraud_dist, 75)
    iqr = q75-q25
    cutoff = iqr*1.5
    upper_limit = q75+cutoff
    lower_limit = q25-cutoff
    outlier_list = [x for x in fraud_dist if x <
                    lower_limit or x > upper_limit]
    nd_dataset = nd_dataset.drop(nd_dataset[(nd_dataset[feature] > upper_limit) | (
        nd_dataset[feature] < lower_limit)].index)
    print(f"Lower limit {lower_limit}")
    print(f"Upper limit {upper_limit}")
    print(outlier_list)
    print(f"Outliers removed {len(outlier_list)}")
    print("\n\n")
# %%
print(f"Remainning instances {len(nd_dataset)}")
# %%
# positive corr new
f, axes = plt.subplots(ncols=4, figsize=(20, 4))
f.suptitle("Positive Correlation")
for i, feature in enumerate(positive_corr):
    sns.boxplot(x="Class", y=feature, data=nd_dataset, ax=axes[i])
    axes[i].set_title("Reduction " + feature)
# %%
# negative corr new
f, axes = plt.subplots(ncols=4, figsize=(20, 4))
f.suptitle("Negative Correlation")
for i, feature in enumerate(negative_corr):
    sns.boxplot(x="Class", y=feature, data=nd_dataset, ax=axes[i])
    axes[i].set_title("Reduction " + feature)
# %%
nd_X = nd_dataset.drop("Class", axis=1)
nd_Y = nd_dataset["Class"]
# %%
X_Tsne = TSNE(n_components=2, random_state=42).fit_transform(nd_X.values)
X_Pca = PCA(n_components=2, random_state=42).fit_transform(nd_X.values)
X_Svd = TruncatedSVD(n_components=2, random_state=42,
                     algorithm="randomized").fit_transform(nd_X.values)
# %%
cluster_list = [X_Tsne, X_Pca, X_Svd]
# %%
f, axes = plt.subplots(ncols=3, figsize=(24, 3))
blue = mpatches.Patch(color="blue", label="normal")
red = mpatches.Patch(color="red", label="fraud")
f.suptitle("Different Clusters")
for i, cluster in enumerate(cluster_list):
    axes[i].scatter(cluster[:, 0], cluster[:, 1], c=(nd_Y == 0),
                    cmap="coolwarm", label="normal", linewidths=2)
    axes[i].scatter(cluster[:, 0], cluster[:, 1], c=(nd_Y == 1),
                    cmap="coolwarm", label="fraud", linewidths=2)
    axes[i].grid(True)
    axes[i].legend(handles=[blue, red])

# %%
