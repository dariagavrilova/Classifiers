import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


# defining data
heart = pd.read_csv('SAHeart.csv', sep=',', header=0)
vowel_train = pd.read_csv('vowel.train.csv', sep=',', header=0)
vowel_test = pd.read_csv('vowel.test.csv', sep=',', header=0)

# function to pass string row to numeric
def to_numeric(row):
    if row['famhist'] == 'Present':
        return 1
    else:
        return 0

# cleaning data
heart.drop(['row.names'], axis=1, inplace=True)
numeric = heart.apply(lambda row: to_numeric(row), axis=1)
heart['famhist_numeric'] = numeric
heart.drop(['famhist'], axis=1, inplace=True)
vowel_train.drop(['row.names'], axis=1, inplace=True)
vowel_test.drop(['row.names'], axis=1, inplace=True)

# splitting data into test y train sets
y = heart.iloc[:, 9]
X = heart.iloc[:, :9]

y_tr = vowel_train.iloc[:,0]
X_tr = vowel_train.iloc[:,1:]
y_test = vowel_test.iloc[:,0]
X_test = vowel_test.iloc[:,1:]

# defining classifiers
LR = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr')
SVM = svm.LinearSVC()
RF = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)


# defining class
class Classifires:
    def __init__(self, data, classifier):
        self.data = data
        self.classifier = classifier

    def train(self, some_X, some_y):
        self.classifier.fit(some_X, some_y)

    def test(self, some_X, some_y):
        self.classifier.predict(some_X.iloc[460:, :])
        print(round(self.classifier.score(some_X, some_y), 2))


p1 = Classifires(heart, LR)
p1.train(X, y)
p1.test(X, y)


p2 = Classifires(heart, SVM)
p2.train(X, y)
p2.test(X, y)


p3 = Classifires(heart, RF)
p3.train(X, y)
p3.test(X, y)


p4 = Classifires(heart, NN)
p4.train(X, y)
p4.test(X, y)


p5 = Classifires(vowel_train, LR)
p5.train(X_tr, y_tr)
p5.test(X_test, y_test)


p6 = Classifires(vowel_train, SVM)
p6.train(X_tr, y_tr)
p6.test(X_test, y_test)


p7 = Classifires(vowel_train, NN)
p7.train(X_tr, y_tr)
p7.test(X_test, y_test)


p8 = Classifires(vowel_train, RF)
p8.train(X_tr, y_tr)
p8.test(X_test, y_test)








