# Credit Card Fraud Prediction

A model to classify wether a credit card transaction is fraudulent or no.

The prediction was done using the Isolation Forest Algorithm.  
It gave a accuracy score of 0.9977177767634563.  
Confusion Matrix  

From the confusion matrix it is clear that the prediction for majority of non fraud transactions is correct while the prediction for the fraud transaction is not. Only approximately 30% of fraudulent transactions are detected. Even though the combined accuracy is very high the goal is to correctly predict the fraud transactions as a wrongly detected fraud transaction .ie. a false negative might be much more harmful.  

## Sub-sampling method  
Since the dataset is imbalanced we will need to perform some balancing techniques so that our model doesn't just assume that it is not a fraud.  
First since the Time and Amount column have not been dimensionally reduced or scaled we scale it using RobustScaler.  
Looking at the heatmap we can infer that V2,V4,V11,V19 are positively correlated while V10,V12,V14,V16 are negatively correlated.  
<!-- <p align="center">
  <img src="./images/heatmap.png" />  
</p>   -->
After removing the outliers 852 instances remain out of 984 this may or may not cause data loss.  
After using a bunch of classifiers we get the cross_val_score as:-
1. Logistic Regression: 89.45%
2. K Nearest Neighbours: 90.62%
3. Support Vector Classifier: 90.05%
4. Decision Tree Classifier: 86.92%

But if we look at the max score instead of the mean we find that Logistic Regression did the best along with KNN at 94.29%  .

After hyperparameter optimization using GridSearchCV the scores are:-
1. Logistic Regression: 95.59%
2. K Nearest Neighbours: 96.18%
3. Support Vector Classifier: 94.71%
4. Decision Tree Classifier: 95.59%

The ROC scores are:-
1. Logistic Regression: 97.81%
2. K Nearest Neighbours: 93.10%
3. Support Vector Classifier: 97.68%
4. Decision Tree Classifier: 94.18%

Looking at the scores we find that KNN give the best accuracy followed by Logistic Regression. But Logistic Regression had the highest ROC score.  
After using undersampling we find that the accuracy score for Logistic regression is only 76%