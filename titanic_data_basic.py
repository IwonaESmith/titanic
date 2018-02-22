
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df_train = pd.read_csv('titanic_train.csv')
df_test = pd.read_csv("titanic_test.csv")


# Exploring data
df_train.describe(include = "all")


    
# Age / sort the ages into logical categories
pd.isnull(df_train["Age"]).sum()

df_train["Age"] = df_train["Age"].fillna(-0.5)
df_test["Age"] = df_test["Age"].fillna(-0.5)
bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
age_group = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
df_train['AgeGroup'] = pd.cut(df_train["Age"], bins, labels = age_group)
df_test['AgeGroup'] = pd.cut(df_test["Age"], bins, labels = age_group)



# Title data
df_all_list = [df_train, df_test]

# extract a title for each name in the train and test datasets
for dataset in df_all_list:
    dataset["Title"] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(df_train['Title'], df_train['Sex'])

#reducing the title data
for dataset in df_all_list:
    dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir','Capt', 'Col',
    'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace(['Mlle', 'Ms'], 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')


df_train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

#factorising title 
title_mapping = {"Mr": 1, "Master": 2, "Mrs": 3, "Miss": 4, "Rare": 5}
df_train['Title'] = df_train['Title'].map(title_mapping)
df_test['Title'] = df_test['Title'].map(title_mapping)

df_train.head()

# Agegroup - title and age (this will estimate age group based on titile)
age_title_mapping = {1: "Young Adult", 2: "Baby", 3: "Adult", 4: "Young Adult", 5: "Adult", 6: "Adult"}
df_train["AgeGroup"] = np.where(df_train['AgeGroup'] == 'Unknown', df_train['Title'].map(age_title_mapping), df_train['AgeGroup'] )
df_test["AgeGroup"] = np.where(df_test['AgeGroup'] == 'Unknown', df_test['Title'].map(age_title_mapping), df_test['AgeGroup'] )
# factorising agegroup 
age_mapping = { 'Unknown':0, 'Baby':1, 'Child':2, 'Teenager':3, 'Student':4, 'Young Adult':5, 'Adult':6, 'Senior':7}
df_train['AgeGroup'] = df_train["AgeGroup"].map(age_mapping)
df_test['AgeGroup'] = df_test["AgeGroup"].map(age_mapping)


# Sex
# factorising sex 
sex_mapping = {'male': 0, 'female': 1}
df_train['Sex'] = df_train['Sex'].map(sex_mapping)
df_test ['Sex']= df_test ['Sex'].map(sex_mapping)


# Embarkment
# factoriisng embarkemnt data
df_train[df_train["Embarked"] == "S"].shape[0]
df_train[df_train["Embarked"] == "C"].shape[0]
df_train[df_train["Embarked"] == "Q"].shape[0]

# most people embarked in southampton hence fill the missing values with (S)
df_train = df_train.fillna({"Embarked": "S"})

embarking_mapping = {"S": 0, "C": 1, "Q":2}
df_train["Embarked"] = df_train["Embarked"].map(embarking_mapping)
df_test["Embarked"] = df_test["Embarked"].map(embarking_mapping)


# Fare
# fare nan value based on the mean value for Pclass in tets set
df_test[[ "Pclass","Fare"]].groupby([ "Pclass"], as_index=False).mean()
df_test = df_test.fillna({"Fare": 12.459678})
# factorising and categorising "Fare"
df_train['FareBand'] = pd.qcut(df_train['Fare'], 4, labels = [1, 2, 3, 4])
df_test['FareBand'] = pd.qcut(df_test['Fare'], 4, labels = [1, 2, 3, 4])


# Cleaning dataset - removing data that is of smallish importance
df_train = df_train.drop([ 'PassengerId', 'Name' , 'Age', 'Ticket','Cabin','Fare'], axis = 1)
df_test = df_test.drop([ 'PassengerId', 'Name' , 'Age', 'Ticket','Cabin','Fare'], axis = 1)


df_train.head()
df_test.head()





#  Data modeling

# Train Test split
df_train.sample(10)

X = df_train.drop("Survived", axis=1)
X['FareBand'] = X['FareBand'].astype(int)
y = df_train["Survived"]


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# building a model 
   # Logistic Regression
   # KNN or k-Nearest Neighbors
   # Support Vector Machines
   # Linear SVC
   # Naive Bayes 
   # Decision Tree
   # Random Forrest
   # Perceptron
   # Stochastic Gradient Descent
   # Gradient Boosting Classifier
   # Xgboost


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix



# Logistic Regression
from sklearn.linear_model import LogisticRegression
logregClassifier = LogisticRegression(random_state = 0)
logregClassifier.fit(X_train, y_train)
y_pred = logregClassifier.predict(X_test)
acc_logregClassifier = round(accuracy_score(y_test, y_pred)*100 , 2)
print(acc_logregClassifier)
cm_logreg = confusion_matrix(y_test, y_pred)


# KNN or k-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
acc_knn = round(accuracy_score(y_test, y_pred)*100 , 2)
cm_knn = confusion_matrix(y_test, y_pred)


# Support Vector Machines 
from sklearn.svm import SVC
SVC = SVC(kernel = 'rbf', random_state = 0)
SVC.fit(X_train, y_train)
y_pred = SVC.predict(X_test)
acc_SVC = round(accuracy_score(y_test, y_pred)*100 , 2)
cm_SVC = confusion_matrix(y_test, y_pred)


# Linear SVC
from sklearn.svm import LinearSVC
linear_svc = LinearSVC()
linear_svc.fit(X_train, y_train)
y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(accuracy_score(y_test, y_pred) * 100, 2)
cm_linear_svc = confusion_matrix(y_test, y_pred)


# Naive Bayes classifier
from sklearn.naive_bayes import GaussianNB
gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
y_pred = gaussian.predict(X_test)
acc_gaussian =  round(accuracy_score(y_test, y_pred) * 100, 2)
cm_gaussian = confusion_matrix(y_test, y_pred)


# Decision Tree
from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(accuracy_score(y_test, y_pred) * 100, 2)
cm_decision_tree = confusion_matrix(y_test, y_pred)

# Extra tree 
from sklearn.ensemble import ExtraTreesClassifier
extra_tree = ExtraTreesClassifier()
extra_tree.fit(X_train, y_train)
y_pred = extra_tree.predict(X_test)
acc_extra_tree = round(accuracy_score(y_test, y_pred) * 100, 2)
cm_extra_tree  = confusion_matrix(y_test, y_pred)


# Random Forest
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)
random_forest.score(X_train, y_train)
acc_random_forest = round(accuracy_score(y_test, y_pred) * 100, 2)
cm_random_forest = confusion_matrix(y_test, y_pred)


# Perceptron
from sklearn.linear_model import Perceptron
perceptron = Perceptron()
perceptron.fit(X_train, y_train)
y_pred = perceptron.predict(X_test)
acc_perceptron = round(accuracy_score(y_test, y_pred) * 100, 2)
cm_perceptron = confusion_matrix(y_test, y_pred)


# Stochastic Gradient Descent
from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier()
sgd.fit(X_train, y_train)
y_pred = sgd.predict(X_test)
acc_sgd = round(accuracy_score(y_test, y_pred) * 100, 2)
cm_sgd = confusion_matrix(y_test, y_pred)


# Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
gbk = GradientBoostingClassifier()
gbk.fit(X_train, y_train)
y_pred = gbk.predict(X_test)
acc_gbk = round(accuracy_score(y_test, y_pred) * 100, 2)
cm_gbk = confusion_matrix(y_test, y_pred)


# XGBoost
from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)
acc_xgb = round(accuracy_score(y_test, y_pred) * 100, 2)
cm_xgb = confusion_matrix(y_test, y_pred)




models = pd.DataFrame({
    'Model': [  'Logistic Regression','KNN','Support Vector Machines', 'Linear SVC' , 'Naive Bayes',  
              'Decision Tree',' Extra tree ','Random Forest','Perceptron', 'Stochastic Gradient Descent' ,
              'Gradient Boosting Classifier',  'XGBoost'],
    'Score': [acc_logregClassifier,  acc_knn,  acc_SVC, acc_linear_svc, acc_gaussian, acc_decision_tree, acc_extra_tree, 
              acc_random_forest, acc_perceptron,  acc_sgd, acc_gbk, acc_xgb ]})
models.sort_values(by='Score', ascending=False)


print(cm_xgb, cm_gbk)



















