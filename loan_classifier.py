# import the necessary packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import pickle

# loading the data
balance_data = pd.read_csv(
    'Decision_Tree_ Dataset.csv', sep=",", header=0)

balance_data.head()


column_mapping = {'1': 'Initial payment',
                  '2': 'Last payment',
                  '3': 'Credit score',
                  '4': "House number",
                  'Unnamed: 5': 'Result'}
# Rename columns using the dictionary
balance_data.rename(columns=column_mapping, inplace=True)
balance_data.head()


# Separating the target varaiable


# Separating the target varaiable

X = balance_data.values[:, 0:4]
Y = balance_data.values[:, 5]
output, uniques = pd.factorize(Y)


# Splitting the datasetinto test and target
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=100)

# Functoin to perform training with entropy
clf_entropy = DecisionTreeClassifier(
    criterion='entropy', random_state=100, max_depth=3, min_samples_leaf=5)
clf_entropy.fit(X_train, y_train)


y_pred = clf_entropy.predict(X_test)
print('Accuracy is', accuracy_score(y_test, y_pred)*100)
classifier_pickle = open('dt.pickle', 'wb')
pickle.dump(clf_entropy, classifier_pickle)
classifier_pickle.close()

output_pickle = open('output.pickle', 'wb')
pickle.dump(uniques, output_pickle)
