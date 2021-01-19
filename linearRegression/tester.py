import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib import style
import pickle


"""
In this project, we will be exploring linear regression, which is essentially a best fit line that is dependent on our input.
We will be using linear regression on our data set to predict the student's final grades.
Linear regression is best used when the data is correlated. 
In our case we will be observing data from the student's past grades, their studytime, their failures, and their absences.
"""

data = pd.read_csv("student-mat.csv", sep=";")
print(data.head())

# trimming the data
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]  # our attributes

predict = "G3"

X = np.array(data.drop([predict], 1))  # features
# we drop our attribute that we want to predict from the data array and in turn
# the rest of our attributes are then, #Features
# feature is an input variable—the x variable in simple linear regression
y = np.array(data[predict])  # label(y), the thing we are trying to predict in our linear regression

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

linear = linear_model.LinearRegression()  #Our model

linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)
print("Our model can predict with accuracy of :", acc)

print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print("computer predicted a grade of ", predictions[x],"using the input values of ", x_test[x],
          "The actual grade is:", y_test[x]) #printing our predictions, our input values, and our actual value