# Import a dataset to work with
from sklearn import datasets
iris = datasets.load_iris()

# Since f(x) = y, we consider x to be the feature here and y to be the label
X = iris.data
Y = iris.target

# Split the dataset into random training and testing subsets (random_state not defined)
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.5)

# To perform data classification using user-chosen classifier
from sklearn import tree
my_classifier = tree.DecisionTreeClassifier()

# Curve-fitting
my_classifier.fit(X_train, Y_train)

# Predicting how accurate original dataset points fit with current classifier
predictions = my_classifier.predict(X_test)

from sklearn.metrics import accuracy_score
print accuracy_score(Y_test, predictions)
