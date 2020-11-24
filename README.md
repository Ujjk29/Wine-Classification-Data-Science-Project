# Wine-Classification-Data-Science-Project
Applying different ML Classification algorithms on the [wine data set](https://archive.ics.uci.edu/ml/datasets/Wine) and getting inferences from the data using Python.

## System Requirements
1. Python3 must be installed on the PC
2. Important statistical libraries such as Numpy, Pandas and scikit-learn.
3. The code can be run in Jupyter-notebook and Google collaboratory both.

## Data Set Information
These data are the results of a chemical analysis of wines grown in the same region in Italy but derived from three different cultivars. The analysis determined the quantities of 13 constituents found in each of the three types of wines. The data is Multivariate.

The attributes are:
1. Alcohol
2. Malic acid 
3. Ash
4. Alkalinity of ash 
5. Magnesium
6. Total phenols
7. Flavanoids
8. Nonflavanoid phenols 
9. Proanthocyanins 
10. Color intensity
11. Hue
12. OD280/OD315 of diluted wines 
13. Proline
![alt text](/Images/Overview%20of%20training%20set%20of%20wine%20dataset.png)

## Attribute and Class Information
There are 13 attributes in total. All attributes are continuous data. Dataset was verified to contain no missing attribute values.
![alt text](/Images/No%20missing%20attribute%20value%20in%20the%20given%20dataset.png)

There are a total of 3 classes and a total of 178 instances:
* Class 1 - 59 instances
* Class 2 - 71 instances
* Class 3 - 48 instances

## Data Preprocessing
We have calculated the measures of central tendencies and it can be summarized in the contingency table below:
![alt text](/Images/Measures%20of%20central%20tendency%20of%20data.png)

## Preliminary Analysis
We can observe that the data is sorted based on class labels. So we infer that we must randomize it before splitting. By observing the data description, we can infer that the features are not closely related to each other. For e.g. Proline values dominate overall central tendency measures over attributes such as Ash content. Hence, we can infer that there is a need for normalizing the attribute values to be contained in a similar domain. We decided to normalize the data after splitting it into test and training set.

## Training Data vs Test Data
Here we are using a simple holdout method where we are keeping 25% of the data for test and 75% for training. The data has been randomized before split as observed in the preliminary stage.
![alt text](/Images/75%25%20of%20the%20data%20is%20training%20dataset.png)
![alt text](/Images/25%25%20of%20the%20dataset%20is%20test%20dataset.png)
![alt text](/Images/Checking%20the%20first%205%20values%20of%20Training%20set%20(for%20randomness).png)
![alt text](/Images/Checking%20the%20first%205%20values%20of%20test%20set.png)

## Normalization of Training and Test Data
We performed two types of scaling:
1. Standard Scaling:
![alt text](/Images/Normalization%20using%20Standard%20scaling.png)
2. Min-Max Scaling:
![alt text](/Images/Training%20Data%20Min-Max%20normalization%20(0-1).png)
We have decided to use Min-Max Scaling over Standard Scaling. Since the values are much closer to each other in min-max and since we know that classifiers such as SVM depend on how good the scaling is performed, min-max dominates over standard scaling.

## Re-analysis of data after partitioning (Training and Test sets) and Normalizing
1. Class Distribution
![alt text](/Images/Class%20distribution%20in%20the%20original%20dataset.png)
![alt text](/Images/Class%20distribution%20in%20training%20dataset.png)
![alt text](/Images/Class%20distribution%20in%20testing%20dataset.png)
We can see that test-class distribution is roughly equivalent in all three datasets. This means accuracy is a good way of measuring classifiers (due to the absence of bias).

2. Class as a function of different attributes
![alt text](/Images/Sharing%20attributes(x)%20per%20class(y).png)
These are the sub-plots of various important attributes and the relation between attribute-values and label classes.
![alt text](/Images/Plot.png)
Attributes and their influence on classification have been calculated in order to drop those attributes which are least important like Nonflavanoid phenols, Ash.

## Classfication and Choosing Appropriate Classifier
We have used SVM (with the linear kernel), Naive Bias Classifiers and Random Forest (decision tree) classifiers.
![alt text](/Images/Test%20result.png)

## Conclusion
Although it may appear counter-intuitive, we conclude naive bias classifiers may be the best classifier in this case. This is because the classifiers are showing extremely high accuracy and we must try to avoid overfitting.

## References
* [SVM](https://scikit-learn.org/stable/modules/svm.html)
* [Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
* [Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html)
* [Matplotlib](https://matplotlib.org/)
* [Feature Selection Techniques in Machine Learning with Python](https://towardsdatascience.com/feature-selection-techniques-in-machine-learning-with-python-f24e7da3f36e)
