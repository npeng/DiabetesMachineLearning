import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
# from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import time
start = time.time()

pd.set_option('max_columns', 999)
pd.set_option('display.width', 1000)

# STEP-1 : READING THE DATA FROM A CSV FILE TO A PANDAS DATAFRAME, DIVIDING IT INTO TRAINING AND TEST SAMPLES AND PREPROCESSING THE TRAINING DATASET
#                'Duration', 'Purpose', 'Class']
column_list = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']


# file_name = inputfile  # filename is argument 1

inputfile = "PimaUpdated.csv"
with open(inputfile, 'rU') as f:  # opens PW fi   le
    reader = csv.reader(f)
    data = list(list(rec) for rec in csv.reader(f, delimiter='\t'))

data=data[1:]
# print("data :",data)
utility_matrix1 = pd.read_csv(inputfile)
# print("utility_matrix1 :",utility_matrix1)

training = utility_matrix1.values.tolist()
X = pd.DataFrame(training)
imputed_data = X

target= imputed_data.as_matrix(columns=['Class'])
# print("target :", target)
# By using the class DaatFrameImputer function from "http://stackoverflow.com/questions/25239958/impute-categorical-missing-values-in-scikit-learn"


# Splitting the dataset into training and testing sets.

#
# train_data_matrix, test_data_matrix = train_test_split(utility_matrix1, test_size=0.2)
#
# train_data = train_data_matrix.values.tolist()
# test_data = test_data_matrix.values.tolist()
# print("len of train data :", len(train_data))
# print("len of test data : ", len(test_data))
#
# train_data_matrix.to_csv('training.csv')
# test_data_matrix.to_csv('testing.csv')

with open("training.csv", 'rU') as f:  # opens PW fi   le
    reader = csv.reader(f)
    train_data = list(list(rec) for rec in csv.reader(f, delimiter='\t'))
# print(train_data)
train_data_matrix_with_outcome = pd.read_csv("training.csv")
# print(train_data_matrix_with_outcome)

with open("testing.csv", 'rU') as f:  # opens PW fi   le
    reader = csv.reader(f)
    test_data = list(list(rec) for rec in csv.reader(f, delimiter='\t'))
# print(test_data)
test_data_matrix_with_outcome = pd.read_csv("testing.csv")
# print("For test.csv :",test_data_matrix_with_outcome)
# print(len(test_data_matrix_with_outcome))


new_header = train_data_matrix_with_outcome.iloc[0] #grab the first row for the header
train_data_matrix = train_data_matrix_with_outcome[1:] #take the data less the header row
train_data_matrix.columns = new_header

# print("train data matrix :",train_data_matrix)

new_header = test_data_matrix_with_outcome.iloc[0] #grab the first row for the header
test_data_matrix = test_data_matrix_with_outcome[1:] #take the data less the header row
test_data_matrix.columns = new_header

# print("test data matrix :", test_data_matrix)




# train_data is the list of 80% dataset
# print("train_data length:", len(train_data))


#
#
#
# # STEP-2 : PERFORMING FEATURE SELECTION VIA PCA
#
# # Normalizing the dataset
#
X_std = StandardScaler().fit_transform(train_data_matrix)
#
#
# print("Normal traing dataset :", X_std)
# # print(len(X_std))
#
Y_std = StandardScaler().fit_transform(test_data_matrix)
# #
# # print("Normal testing dataset :", Y_std)
# # print(len(Y_std))
#
#
# # Eigen decomposition of the standardized data based on the correlation matrix:
#

cov_mat = np.cov(X_std.T)
# print("Covariance Matrix :", cov_mat)
#
#
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
#
# # print("eig_vals :", eig_vals)
# # print("eig_vecs :", eig_vecs)
#
#
#
# # Performing SVD
u,s,v = np.linalg.svd(X_std.T)
#
# # Comparing the results of eigen decomposition of covariance and correlation matrix with SVD we find that it is the same.
#
#
# # Now finding the number of principal components for PCA.
#
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
#
#
# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort()
eig_pairs.reverse()
#
#
#
# How to select the number of principal components
tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
#
# # Observing the eigen values we can keep all the fetures in our dtaset as they do have a significant
# # contribution to our predictions for the dataset.
#
# # Creating the target variable
target_train_find= train_data_matrix_with_outcome.as_matrix(columns=['Outcome'])
target_test_find = test_data_matrix_with_outcome.as_matrix(columns=['Outcome'])
# # print("target :",target_train)
#
train_data_matrix_with_outcome.drop('Outcome', axis=1, inplace=True)
test_data_matrix_with_outcome.drop('Outcome', axis=1, inplace=True)
# # print(train_data_matrix_with_outcome)
#
#
# # Converting the data as numpy array
data1 = np.array(train_data_matrix_with_outcome)
# print("numpy aray of data1 without outcome :", data1)
data2 = np.array(test_data_matrix_with_outcome)
# print("numpy aray of data2 without outcome :", data2)


data_train = data1
data_test = data2
target_train = target_train_find
target_test = target_test_find

# print("target_test :", target_test)
#
# print(data_train.shape)
# print(target_train.shape)
# print(target_test.shape)
# print(data_test.shape)
#
#
c, r = target_train_find.shape
target_train_find = target_train_find.reshape(c,)

c, r = target_test_find.shape
target_test_find = target_test_find.reshape(c,)

#
# # Performing Logistic Regression by using the inbuilt functions from sklearn library. I took the function from
# # "http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html" and modified it according to my needs.
print("Logistic Regression")
logreg =LogisticRegression()
logreg.fit(data_train,target_train)
y_pred_logreg=logreg.predict(data_test)
target_names=['YES', 'NO']

print("Accuracy is :",accuracy_score(target_test, y_pred_logreg))
print(classification_report(target_test, y_pred_logreg, target_names=target_names))

cf_matrix_logreg = confusion_matrix(target_test, y_pred_logreg)
#plt.matshow(cf_matrix_logreg)
#plt.title('Confusion Matrix for Logistic Regression')
#plt.colorbar()
#plt.ylabel('True label')
#plt.xlabel('Predicted label')
#plt.show()


# #########################################################################

# Performing the Ada Boosting Classifier by using "http://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_twoclass.html"
# and make necessary modifications to it and moulded it as per my needs.
print("Ada Boosting")
abc=AdaBoostClassifier(n_estimators=100)
abc.fit(data_train,target_train)
y_pred_abc=abc.predict(data_test)

print("Accuracy is :",accuracy_score(target_test, y_pred_abc))
print(classification_report(target_test, y_pred_abc, target_names=target_names))


cf_matrix_adaboosting = confusion_matrix(target_test, y_pred_abc)

#plt.matshow(cf_matrix_adaboosting)
#plt.title('Confusion Matrix for Ada Boosting')
#plt.colorbar()
#plt.ylabel('True label')
#plt.xlabel('Predicted label')
#plt.show()
# ##############################################################################

# Performing Naive Bayes Classifier by using "http://scikit-learn.org/stable/modules/naive_bayes.html" and using it in my code to
# check the performance of Bayes MinimumClassifier.
print("Bayes Minimum Classifier")
gnb=GaussianNB()
gnb.fit(data_train,target_train)
y_pred_gnb=gnb.predict(data_test)

print("Accuracy is :",accuracy_score(target_test, y_pred_gnb))
print(classification_report(target_test, y_pred_gnb, target_names=target_names))
cf_matrix_bayes = confusion_matrix(target_test, y_pred_gnb)

#plt.matshow(cf_matrix_bayes)
#plt.title('Confusion Matrix for Naive Bayes')
#plt.colorbar()
#plt.ylabel('True label')
#plt.xlabel('Predicted label')
#plt.show()


# #######################################################################################

# Perfroming Random Forest technique by using the inbuilt function from "http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html"
# and modified it accordingly to our dataset.
print("Random Forest")
rnd_fc=RandomForestClassifier(n_estimators=50,criterion='entropy')
rnd_fc.fit(data_train,target_train)
y_pred_rnd_fc=rnd_fc.predict(data_test)


print("Accuracy is :",accuracy_score(target_test, y_pred_rnd_fc))
print(classification_report(target_test, y_pred_rnd_fc, target_names=target_names))

cf_matrix_rndf = confusion_matrix(target_test, y_pred_gnb)

#plt.matshow(cf_matrix_rndf)
#plt.title('Confusion Matrix for Random Forest')
#plt.colorbar()
#plt.ylabel('True label')
#plt.xlabel('Predicted label')
#plt.show()


################################################################################
end = time.time()
print("Execution time: "),
print(end - start),
print("seconds")
