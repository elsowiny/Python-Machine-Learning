import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

data = pd.read_csv("student-mat.csv", sep=";")
print(data.head())

# trimming the data
data = data[["G1", "G2", "G3", "studytime","failures","absences"]] #our attributes

predict = "G3"

X = np.array(data.drop([predict], 1)) #features
# we drop our attribute that we want to predict from the data array and in turn
# the rest of our attributes are then, #Features
#feature is an input variable—the x variable in simple linear regression
y = np.array(data[predict]) #label(y), the thing we are trying to predict in our linear regression

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)