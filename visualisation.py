# %%
import numpy as np
import pandas as pd
import sklearn
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
# %%
RANDOM_SEED = 42
LABELS = ["normal", "fraud"]

# %%
dataset = pd.read_csv('./dataset/creditcard.csv')
dataset.head()
# %%
dataset.info()
# %%
classes = pd.value_counts(dataset['Class'], sort=True)
classes.plot(kind='bar')
plt.title("Transaction Distribution")
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.xticks(range(2), LABELS)
# %%
fraud = dataset[dataset["Class"] == 1]
normal = dataset[dataset["Class"] == 0]
print(fraud.shape, normal.shape)

# %%
fraud.Amount.describe()
# %%
normal.Amount.describe()
# %%
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle("Transaction Amount")
bins = 50
ax1.hist(fraud.Amount, bins=bins)
ax1.set_title("Fraud")
ax2.hist(normal.Amount, bins=bins)
ax2.set_title("Normal")
plt.xlabel("Amount")
plt.ylabel("Frequency")
plt.yscale('log')
plt.show()
# %%
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle("Transaction Amount VS Transaction Time")
ax1.scatter(fraud.Time, fraud.Amount)
ax1.set_title("Fraud")
ax2.scatter(normal.Time, normal.Amount)
ax2.set_title("Normal")
plt.xlabel("Time")
plt.ylabel("Amount")
plt.show()
# %%
dataset_sample = dataset.sample(frac=0.1, random_state=RANDOM_SEED)
dataset_sample.shape
# %%
corrmat = dataset_sample.corr()
top_features = corrmat.index
g = sns.heatmap(dataset_sample[top_features].corr(), cmap="RdYlGn")
# %%
columns = dataset_sample.columns.tolist()
columns = [c for c in columns if c not in ["Class"]]
target = "Class"
# %%
state = np.random.RandomState(RANDOM_SEED)
X = dataset_sample[columns]
Y = dataset_sample[target]

# %%
fraud_sample = dataset_sample[dataset_sample["Class"] == 1]
normal_sample = dataset_sample[dataset_sample["Class"] == 0]
outlier_fraction = len(fraud_sample)/float(len(normal_sample))
print(outlier_fraction)
# %%
IFclassifier = IsolationForest(n_estimators=100, max_samples=len(
    X), contamination=outlier_fraction, random_state=state)
y_pred = IFclassifier.fit_predict(X)
scores = IFclassifier.decision_function(X)
y_pred[y_pred == 1] = 0
y_pred[y_pred == -1] = 1
n_errors = (y_pred != Y).sum()
# %%
print(accuracy_score(Y, y_pred))
print(classification_report(Y, y_pred))
# %%
