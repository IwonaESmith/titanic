

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

#load data


df_train = pd.read_csv('titanic_train.csv')
df_test = pd.read_csv("titanic_test.csv")
df_all = pd.concat([df_train, df_test] )

# exploring data

pd.show_versions()

df.columns

print("train", df_train.shape )
print("test", df_test.shape )

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
from sklearn.linear_model import LogisticRegression
#linear classifier with stochastic gradient descent (gradient of loss function)
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_validate

import xgboost as xgb

df_train.sample(10)





# selecting int values
df_train.select_dtypes(include =[np.int, np.float]).head()
def get_feats(df):
    feats = df.select_dtypes(include=[np.int]).columns.values
    black_list = ["Passanger", "Survived"]
    return [feat for feat in feats if feat not in black_list]
#changing string values to int
def feature_engeneering(df):
    df['sex_cat'] = pd.factorize(df_all["Sex"])[0]
    df['embarked_cat'] = pd.factorize(df_all["Embarked"])[0]
#call fro models
def get_models():
    return[
            ('lr', LogisitcRegression()),
            ('dt', DecisionTreeClassifier()),
            ('rf', RandomForestClassifier()),
            ('et', ExtraTreeClassifier())
            ]
# visalusation
def plot_result(model_name, result, ylim=(0,1.)):
    mean_train = np.round(np.mean(result['train_score']), 2)
    mean_test = np.round(np.mean(result['test_score']), 2)

    plt.plot(result['train_score'], 'r-o', label='train')
    plt.plot(result['train_score'], 'g-o', label='test')
    plt.legend(loc='best')
    plt.ylabel("accuracy")
    plt.xlabel('# of fold')
    plt.ylim(*ylim)
    plt.show()
    
# crossvalidation
df = feature_engeneering(df_train)
get_feats(df)
X = df_train[get_feats(df_train)].values
y = df_train["survived"].values



df_all['Fare'].hist(bins=100)
df_all[df_all.Ticket == 'CA.2343']