

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



df_train = pd.read_csv('titanic_train.csv')
df_test = pd.read_csv("titanic_test.csv")
df_all = pd.concat([df_train, df_test] )

# exploring data

pd.show_versions()

df.columns
df_train.shape 
df_test.shape 

passanger = df_train.shape[0] + df_test.shape[0]
print(passanger)

df_train.info()
df_train.head()

df_train['Sex'].value_counts()
df_train['Survived'].value_counts()

data_survived_gr= df_train.groupby([ "Pclass","Sex", "Survived"])

data_survived_gr.size()

data_survived_gr1= df_train.groupby([ "Pclass","Sex"])[ "Survived"].value_counts(normalize=True)
print(data_survived_gr1)

# comapre train && test by sex
describe_fields = ["Age", "Fare", "Pclass", "SibSp", "Parch"]
print(df_train[df_train.Sex == "male"][describe_fields].describe())
print(df_test[df_test.Sex == "male"][describe_fields].describe())
print(df_train[df_train.Sex == "female"][describe_fields].describe())
print(df_test[df_test.Sex == "female"][describe_fields].describe())



# building first (dummy) model to predict survival 
df_train.sample(10)

X_train = df_train.iloc[:,[2]].values
y_train = df_train.iloc[:,1].values

# Fitting dummy classifirer to the Training set
from sklearn.dummy import DummyClassifier
classifier = DummyClassifier()
classifier.fit(X_train,y_train)

# Predicting the results for dummy
y_pred = classifier.predict(X_train)


from sklearn.metrics import accuracy_score
score = accuracy_score(y_train, y_pred)
print( "score: %.2f" % score)

# building 

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

np.random.seed(2018)


df_train.sample(10)





