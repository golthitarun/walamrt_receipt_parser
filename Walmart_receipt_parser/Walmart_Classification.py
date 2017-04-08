
# coding: utf-8

# In[2]:

import pandas as pd
import numpy as np
import sys,os
from difflib import get_close_matches


# In[3]:

# Method used to find a word in the text file with close match to a key. 

def fuzzy_check(data, key, accuracy = 0.6):
    for line in data:
        words = line.lower().split()
        #print words
        flag = get_close_matches(key, words, 1, accuracy)
        if flag:
            return True
    return False


# In[17]:

#Method used to check if walmart is present in the text file or not

def check_walmart(row):
    path = os.path.join(src,row['EXT_ID']+".txt")
    with open(path) as f:
                data = [line.rstrip('\n') for line in f]
                for line in data:
                    #print line.lower()
                    for key in walmart_key:
                        if key in line.lower():
                            return 1
                    
                for key in walmart_key:
                    check = fuzzy_check(data, key, 0.6)
                    if(check):
                        return 1
                return 0
    


# In[74]:

# helper method for checking presence of other words in the text file.
def helper(data, key):
    check = fuzzy_check(data,key, 0.6)
    if(check):
        return 1
    else:
        return 0


# In[75]:

#Method to check the presence of word save in text file.
def check_save(row):
    path = os.path.join(src,row['EXT_ID']+".txt")
    with open(path) as f:
        data = [line.rstrip('\n') for line in f]
        return helper(data, 'save')
                   


# In[76]:

#Method to check the presence of word money in text file.
def check_money(row):
    path = os.path.join(src,row['EXT_ID']+".txt")
    with open(path) as f:
        data = [line.rstrip('\n') for line in f]
        return helper(data, 'money')


# In[77]:

#Method to check the presence of word live in text file.
def check_live(row):
    path = os.path.join(src,row['EXT_ID']+".txt")
    with open(path) as f:
        data = [line.rstrip('\n') for line in f]
        return helper(data, 'live')


# In[78]:

#Method to check the presence of word better in text file.
def check_better(row):
    path = os.path.join(src,row['EXT_ID']+".txt")
    with open(path) as f:
        data = [line.rstrip('\n') for line in f]
        return helper(data, 'better')


# In[79]:

#Method to check the presence of word target in text file.
def check_target(row):
    path = os.path.join(src,row['EXT_ID']+".txt")
    with open(path) as f:
        data = [line.rstrip('\n') for line in f]
        return helper(data, 'target')


# In[104]:

src = "/home/tharunn/Documents/project/train"
walmart_key = ['walmart', 'wa!mar!', 'wal mart', 'Walm0rt', 'walm art', 'HUalmart']
#target_key = ['target']
moto = ['save','money','live','better']
train = pd.read_csv("/home/tharunn/Documents/project/training_data.csv")
has_walmart = train.apply(check_walmart, axis=1, raw=True)
#print has_walmart
ftr_train = pd.DataFrame()
ftr_train['has_walmart'] = has_walmart
ftr_train['has_save'] = train.apply(check_save, axis=1, raw=True)
ftr_train['has_money'] = train.apply(check_money, axis=1, raw=True)
ftr_train['has_live'] = train.apply(check_live, axis=1, raw=True)
ftr_train['has_better'] = train.apply(check_better, axis=1, raw=True)
#ftr_train['has_target'] = train.apply(check_target, axis=1, raw=True)
ftr_train['is_walmart'] = train['IsWalmart'].values



# In[105]:

# Initial extraction of features without including the feature of having target.
ftr_train


# In[119]:

ftr_train_target = pd.DataFrame()
ftr_train_target['has_walmart'] = has_walmart
ftr_train_target['has_save'] = ftr_train['has_save']
ftr_train_target['has_money'] = ftr_train['has_money']
ftr_train_target['has_live'] = ftr_train['has_live']
ftr_train_target['has_better'] = ftr_train['has_better']
ftr_train_target['has_target'] = train.apply(check_target, axis=1, raw=True)
ftr_train_target['is_walmart'] = train['IsWalmart'].values
# features along with the target file being included. 
ftr_train_target


# In[120]:

# Splitting the training data into train and validation so as to check my accuracy.
train_1 =ftr_train.sample(frac = 0.7)
test_1 = ftr_train.loc[~ftr_train.index.isin(train_1.index)]
ftr_train.to_csv('out.csv')


# In[122]:

#Importing different classifiers and splitting the feature and class colums from the data. 

from sklearn.metrics import accuracy_score, log_loss
from sklearn import tree
from sklearn import linear_model
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
x_train = train_1[train_1.columns[0:5]]
y_train  = train_1[train_1.columns[5]]
x_test = test_1[test_1.columns[0:5]]
y_test = test_1[test_1.columns[5]]

x_test


# In[123]:

# Applying different classifiers in my features to check which one is working the best. 

clf = tree.DecisionTreeClassifier()
dt_clf = clf.fit(x_train,y_train)
dt_predict = dt_clf.predict(x_test)
dt_acc = accuracy_score(y_test,dt_predict)
print 'DecisionTree accuracy :', dt_acc

clf = linear_model.Perceptron()
pt_clf = clf.fit(x_train,y_train)
pt_predict = pt_clf.predict(x_test)
pt_acc = accuracy_score(y_test,pt_predict)
print 'Perceptron accuracy :',pt_acc


clf = MLPClassifier()
nn_clf = clf.fit(x_train,y_train)
nn_predict = nn_clf.predict(x_test)
nn_acc = accuracy_score(y_test,nn_predict)
print 'Neural Net accuracy :',nn_acc

clf = svm.SVC()
svm_clf = clf.fit(x_train,y_train)
svm_predict = svm_clf.predict(x_test)
svm_acc = accuracy_score(y_test,svm_predict)
print 'SVM accuracy :',svm_acc


clf = GaussianNB()
nb_clf = clf.fit(x_train,y_train)
nb_predict = nb_clf.predict(x_test)
nb_acc = accuracy_score(y_test,nb_predict)
print 'naïve Bayes accuracy :',nb_acc


clf = LogisticRegression()
lr_clf = clf.fit(x_train,y_train)
lr_predict = lr_clf.predict(x_test)
lr_acc = accuracy_score(y_test,lr_predict)
print 'Logistic Regression accuracy :',lr_acc


clf = KNeighborsClassifier(6)
knn_clf = clf.fit(x_train,y_train)
knn_predict = knn_clf.predict(x_test)
knn_acc = accuracy_score(y_test,knn_predict)
print 'k-Nearest Neighbors accuracy :',knn_acc


clf = BaggingClassifier()
bg_clf = clf.fit(x_train,y_train)
bg_predict = bg_clf.predict(x_test)
bg_acc = accuracy_score(y_test,bg_predict)
print 'Bagging accuracy :',bg_acc


clf = RandomForestClassifier()
rf_clf = clf.fit(x_train,y_train)
rf_predict = rf_clf.predict(x_test)
rf_acc = accuracy_score(y_test,rf_predict)
print 'Random Forests :',rf_acc


clf = AdaBoostClassifier()
ada_clf = clf.fit(x_train,y_train)
ada_predict = ada_clf.predict(x_test)
ada_acc = accuracy_score(y_test,ada_predict)
print 'AdaBoost accuracy :',ada_acc


clf = GradientBoostingClassifier()
gb_clf = clf.fit(x_train,y_train)
gb_predict = gb_clf.predict(x_test)
gb_acc = accuracy_score(y_test,gb_predict)
print 'Gradient Boosting accuracy :',gb_acc
        


# In[124]:

# Splitting the train and validation data for feature set including target. 

train_1 =ftr_train_target.sample(frac = 0.7)
test_1 = ftr_train_target.loc[~ftr_train.index.isin(train_1.index)]


# In[126]:

#splitting the class label and features 

x_train = train_1[train_1.columns[0:6]]
y_train  = train_1[train_1.columns[6]]
x_test = test_1[test_1.columns[0:6]]
y_test = test_1[test_1.columns[6]]
x_train


# In[127]:

#Applying different classifiers on the data including target.

clf = tree.DecisionTreeClassifier()
dt_clf = clf.fit(x_train,y_train)
dt_predict = dt_clf.predict(x_test)
dt_acc = accuracy_score(y_test,dt_predict)
print 'DecisionTree accuracy :', dt_acc

clf = linear_model.Perceptron()
pt_clf = clf.fit(x_train,y_train)
pt_predict = pt_clf.predict(x_test)
pt_acc = accuracy_score(y_test,pt_predict)
print 'Perceptron accuracy :',pt_acc


clf = MLPClassifier()
nn_clf = clf.fit(x_train,y_train)
nn_predict = nn_clf.predict(x_test)
nn_acc = accuracy_score(y_test,nn_predict)
print 'Neural Net accuracy :',nn_acc

clf = svm.SVC()
svm_clf = clf.fit(x_train,y_train)
svm_predict = svm_clf.predict(x_test)
svm_acc = accuracy_score(y_test,svm_predict)
print 'SVM accuracy :',svm_acc


clf = GaussianNB()
nb_clf = clf.fit(x_train,y_train)
nb_predict = nb_clf.predict(x_test)
nb_acc = accuracy_score(y_test,nb_predict)
print 'naïve Bayes accuracy :',nb_acc


clf = LogisticRegression()
lr_clf = clf.fit(x_train,y_train)
lr_predict = lr_clf.predict(x_test)
lr_acc = accuracy_score(y_test,lr_predict)
print 'Logistic Regression accuracy :',lr_acc


clf = KNeighborsClassifier(6)
knn_clf = clf.fit(x_train,y_train)
knn_predict = knn_clf.predict(x_test)
knn_acc = accuracy_score(y_test,knn_predict)
print 'k-Nearest Neighbors accuracy :',knn_acc


clf = BaggingClassifier()
bg_clf = clf.fit(x_train,y_train)
bg_predict = bg_clf.predict(x_test)
bg_acc = accuracy_score(y_test,bg_predict)
print 'Bagging accuracy :',bg_acc


clf = RandomForestClassifier()
rf_clf = clf.fit(x_train,y_train)
rf_predict = rf_clf.predict(x_test)
rf_acc = accuracy_score(y_test,rf_predict)
print 'Random Forests :',rf_acc


clf = AdaBoostClassifier()
ada_clf = clf.fit(x_train,y_train)
ada_predict = ada_clf.predict(x_test)
ada_acc = accuracy_score(y_test,ada_predict)
print 'AdaBoost accuracy :',ada_acc


clf = GradientBoostingClassifier()
gb_clf = clf.fit(x_train,y_train)
gb_predict = gb_clf.predict(x_test)
gb_acc = accuracy_score(y_test,gb_predict)
print 'Gradient Boosting accuracy :',gb_acc
        


# In[130]:

#Extracting features from the test data set.

src = "/home/tharunn/Documents/project/test"
test = pd.read_csv("/home/tharunn/Documents/project/test_data.csv")
ftr_test = pd.DataFrame()
ftr_test['has_walmart'] = test.apply(check_walmart, axis=1, raw=True)
ftr_test['has_save'] = test.apply(check_save, axis=1, raw=True)
ftr_test['has_money'] = test.apply(check_money, axis=1, raw=True)
ftr_test['has_live'] = test.apply(check_live, axis=1, raw=True)
ftr_test['has_better'] = test.apply(check_better, axis=1, raw=True)
ftr_test


# In[133]:

#splitting the training data into class label and features. 
x_train = ftr_train[ftr_train.columns[0:5]]
y_train  = ftr_train[ftr_train.columns[5]]
x_train


# In[141]:

# Applying the SVM classifier since we find it to be having the highest accuracy over validation data.  
clf = svm.SVC()
svm_clf = clf.fit(x_train,y_train)
svm_predict = svm_clf.predict(ftr_test)
confidence_Score = svm_clf.decision_function(ftr_test)
svm_predict


# In[162]:

# helper method to update the output file
def update_result(row):
    if row['IsWalmart'] == 0:
        return "FALSE"
    else:
        return "TRUE"


# In[163]:

# Writing the results to a output csv file. 

result = pd.DataFrame()
result['EXT_ID'] = test['EXT_ID']
result['IsWalmart'] = svm_predict
result['IsWalmart'] = result.apply(update_result, axis=1, raw=True)
result['Confidence Score'] = confidence_Score
result.to_csv("result.csv")


# In[ ]:



