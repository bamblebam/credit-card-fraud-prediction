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
