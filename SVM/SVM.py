import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

cancer = datasets.load_breast_cancer()

# print(cancer.feature_names)
# print(cancer.target_names)

x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)
# print(x_train[:5], y_train[:5])  # To view our data


# print(x_train, y_train)


#classifier = svm.SVC(kernel="poly", degree=2)
classifier = svm.SVC(kernel="linear", degree=2)
""" We use the kernel param for our Support vector classification, as otherwise
would lead to just random guessing and bad accuracy """
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)  # Predict values for our test data

acc = metrics.accuracy_score(y_test, y_pred)  # Test them against our correct values

print(acc)
