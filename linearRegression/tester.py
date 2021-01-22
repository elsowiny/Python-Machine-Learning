import pandas as pd
import numpy as np
from sklearn import linear_model
import sklearn
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib import style
import pickle


"""
In this project, we will be exploring linear regression.
Linear Regression is essentially a best fit line that is dependent on our input.
We will be using linear regression on our data set to predict the student's final grades.
Linear regression is best used when the data is correlated. 
In our case we will be observing data from the student's past grades. 
Such as: their study time, their failures, and their absences.
"""

data = pd.read_csv("student-mat.csv", sep=";")
print(data.head())

# trimming the data
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]  # our attributes

predict = "G3"

x = np.array(data.drop([predict], 1))  # features
# we drop our attribute that we want to predict from the data array and in turn
# the rest of our attributes are then, #Features
# feature is an input variable—the x variable in simple linear regression
y = np.array(data[predict])  # label(y), the thing we are trying to predict in our linear regression

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

# Now we will train our model x number of times, to get our best prediction
# We can change the var numberOfTrainings to decide how many iterations to try for
# An acc of 85% is good, an acc of 95% is even better


# Commented out the first go around of model training.
# The model I have found to be best from the code below is included as a pickle file in this directory
"""
best = 0
numberOfTrainings = 20
for _ in range(numberOfTrainings):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
    linear = linear_model.LinearRegression()  # We create a new model named linear
    linear.fit(x_train, y_train)  # Finding the best fit
    acc = linear.score(x_test, y_test)  # our accuracy
    print("This model has an accuracy of: ", acc)

    if acc > best:
        best = acc  # Change our target accuracy score for each subsequent model to achieve
        with open("finalGradePredictModel.pickle", "wb") as f:
            pickle.dump(linear, f)  # We save the best model until a better one comes along (while we're in the loop)
"""


# Now we must load our model

pickle_in = open("finalGradePredictModel.pickle", "rb")  # We open our pickle file containing our best model
linear = pickle.load(pickle_in)  # Now we load up our model and get ready to use it

print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print("computer predicted a grade of ", predictions[x],"using the input values of ", x_test[x],
          "The actual grade is:", y_test[x])  # Printing our predictions, our input values, and our actual value


plot = "G1"  # One of our attributes
plt.scatter(data[plot], data["G3"])
plt.legend(loc=4)
plt.xlabel(plot)
plt.ylabel("Final Grade")
plt.show()
