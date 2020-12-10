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
nd_Xtrain, nd_Xtest, nd_Ytrain, nd_Ytest = train_test_split(
    nd_X, nd_Y, random_state=42, test_size=0.2)
nd_Xtrain = nd_Xtrain.values
nd_Xtest = nd_Xtest.values
nd_Ytrain = nd_Ytrain.values
nd_Ytest = nd_Ytest.values
# %%
classifiers = {
    "Logistic Reg": LogisticRegression(),
    "K nearest": KNeighborsClassifier(),
    "SVC": SVC(),
    "Decision tree": DecisionTreeClassifier()
}
# %%
for key, value in classifiers.items():
    value.fit(nd_Xtrain, nd_Ytrain)
    score = cross_val_score(value, nd_Xtest, nd_Ytest, cv=5)
    print(f"{key} - {round(score.mean(),4)} - {round(score.max(),4)}")
# %%
log_reg_params = {
    "penalty": ["l1", "l2"],
    "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000]
}
knearest_params = {
    "n_neighbors": list(range(2, 5, 1)),
    "algorithm": ["auto", "ball_tree", "kd_tree", "brute"]
}
svc_params = {
    "C": [0.5, 0.7, 0.9, 1],
    "kernel": ["rbf", "poly", "sigmoid", "linear"]
}
decision_tree_params = {
    "criterion": ["gini", "entropy"],
    "max_depth": list(range(2, 5, 1)),
    "min_samples_leaf": list(range(5, 8, 1))
}

# %%
grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params)
grid_knearest = GridSearchCV(KNeighborsClassifier(), knearest_params)
grid_svc = GridSearchCV(SVC(), svc_params)
grid_decision_tree = GridSearchCV(
    DecisionTreeClassifier(), decision_tree_params)
grid_log_reg.fit(nd_Xtrain, nd_Ytrain)
grid_knearest.fit(nd_Xtrain, nd_Ytrain)
grid_svc.fit(nd_Xtrain, nd_Ytrain)
grid_decision_tree.fit(nd_Xtrain, nd_Ytrain)
# %%
log_reg = grid_log_reg.best_estimator_
knearest = grid_knearest.best_estimator_
svc = grid_svc.best_estimator_
decision_tree = grid_log_reg.best_estimator_
# %%
print(f"log reg - {log_reg.score(nd_Xtrain, nd_Ytrain)}")
print(f"k nearest - {knearest.score(nd_Xtrain, nd_Ytrain)}")
print(f"SVC - {svc.score(nd_Xtrain, nd_Ytrain)}")
print(f"decision tree - {decision_tree.score(nd_Xtrain, nd_Ytrain)}")

# %%
f, axes = plt.subplots(ncols=2, nrows=2)
f.suptitle("Confusion Matrices")
plot_confusion_matrix(log_reg, nd_Xtest, nd_Ytest, ax=axes[0][0])
axes[0][0].set_title("Logistic Regression")
plot_confusion_matrix(knearest, nd_Xtest, nd_Ytest, ax=axes[0][1])
axes[0][1].set_title("KNN")
plot_confusion_matrix(svc, nd_Xtest, nd_Ytest, ax=axes[1][0])
axes[1][0].set_title("SVC")
plot_confusion_matrix(decision_tree, nd_Xtest, nd_Ytest, ax=axes[1][1])
axes[1][1].set_title("Decision Tree")

# %%
log_reg_pred = cross_val_predict(
    log_reg, nd_Xtrain, nd_Ytrain, cv=5, method="decision_function")
knearest_pred = cross_val_predict(knearest, nd_Xtrain, nd_Ytrain, cv=5)
svc_pred = cross_val_predict(
    svc, nd_Xtrain, nd_Ytrain, cv=5, method="decision_function")
decision_tree_pred = cross_val_predict(
    decision_tree, nd_Xtrain, nd_Ytrain, cv=5)

# %%
print(f"Logistic Regression {roc_auc_score(nd_Ytrain,log_reg_pred)}")
print(f"KNN {roc_auc_score(nd_Ytrain,knearest_pred)}")
print(f"SVC {roc_auc_score(nd_Ytrain,svc_pred)}")
print(f"Decision Tree {roc_auc_score(nd_Ytrain,decision_tree_pred)}")

# %%
X_undersample = dataset.drop("Class", axis=1)
Y_undersample = dataset["Class"]
# %%
for train_index, test_index in SKfold.split(X_undersample, Y_undersample):
    X_undersample_train, X_undersample_test = X_undersample.iloc[
        train_index], X_undersample.iloc[test_index]
    Y_undersample_train, Y_undersample_test = Y_undersample.iloc[
        train_index], Y_undersample.iloc[test_index]
X_undersample_train = X_undersample_train.values
X_undersample_test = X_undersample_test.values
Y_undersample_train = Y_undersample_train.values
Y_undersample_test = Y_undersample_test.values
# %%
undersample_accuracy = []
undersample_precision = []
undersample_recall = []
undersample_f1 = []
undersample_auc = []
# %%
for train_index, test_index in SKfold.split(X_undersample_train, Y_undersample_train):
    undersample_pipeline = imbalanced_make_pipeline(
        NearMiss(sampling_strategy="majority"), log_reg)
    undersample_model = undersample_pipeline.fit(
        X_undersample_train[train_index], Y_undersample_train[train_index])
    undersample_prediction = undersample_model.predict(
        X_undersample_train[test_index])
    undersample_accuracy.append(undersample_pipeline.score(
        og_X_train[test_index], og_Y_train[test_index]))
    undersample_precision.append(precision_score(
        Y_undersample_train[test_index], undersample_prediction))
    undersample_recall.append(recall_score(
        Y_undersample_train[test_index], undersample_prediction))
    undersample_f1.append(
        f1_score(Y_undersample_train[test_index], undersample_prediction))
    undersample_auc.append(roc_auc_score(
        Y_undersample_train[test_index], undersample_prediction))

# %%
log_reg_Y_pred = log_reg.predict(nd_Xtrain)

# %%
print("non cross val values for log reg")
print(f"Recall score {recall_score(nd_Ytrain,log_reg_Y_pred)}")
print(f"Precision score {precision_score(nd_Ytrain,log_reg_Y_pred)}")
print(f"F1 score {f1_score(nd_Ytrain,log_reg_Y_pred)}")
print(f"Accuracy score {accuracy_score(nd_Ytrain,log_reg_Y_pred)}")
# %%
print("cross val values for log reg")
print(f"Recall score {np.mean(undersample_recall)}")
print(f"Precision score {np.mean(undersample_precision)}")
print(f"F1 score {np.mean(undersample_f1)}")
print(f"Accuracy score {np.mean(undersample_accuracy)}")
# %%
