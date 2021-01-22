import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

"""
In this example of ML, we use K nearest neighbors to help classify our data.
K nearest neighbors works by looking at the k closest data points that surround our data point that we
want to classify. Our algorithm will look at what class occurs the most that is closest, to be our predicted value.


This algorithm is computationally heavy and is best used on small data sets with few features/dimensions 
"""

data = pd.read_csv("car.data")
print(data.head())

# We convert our string data
le = preprocessing.LabelEncoder()

# The method fit_transform() takes a list (each of our columns) and will return to us an array containing our new values
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

predict = "class"

X = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

# This is going to be how many points we should use to make our decision.
# The K closest points to our data point.
n_neighbors_we_look_at = 7
model = KNeighborsClassifier(n_neighbors=n_neighbors_we_look_at)
# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

model.fit(x_train, y_train)  # Training our model
acc = model.score(x_test, y_test)
print(acc)

predicted = model.predict(x_test)
names = ['acc', 'good', 'unacc', 'vgood']
#  We create a names list so that we can convert our integer predictions into
#  their string representation
# This following code will display, our data our algo prediction on the class, the actual class, and if it was correct

total_predicted = 0
total = 0
for x in range(len(predicted)):
    print("Using data of: ", x_test[x], "\n"
                                        "The algo predicted: ", names[predicted[x]], "\n"
                                                                                     "Actual: ", names[y_test[x]])
    if names[predicted[x]] == names[y_test[x]]:
        print("The model classified CORRECTLY")
        total_predicted += 1
    else:
        print("The model classified INCORRECTLY")
    total += 1

    # n = neighbors of each point in our testing data
    n = model.kneighbors([x_test[x]], 9, True)
    print("N: ", n)
    print("\n")

print("Computer predicted: ", total_predicted, "out of:", total)
print(total_predicted / total * 100, "accuracy rating")
print("Viewing ", n_neighbors_we_look_at, " neighbors")