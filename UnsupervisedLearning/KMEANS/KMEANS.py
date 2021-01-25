import numpy as np
import sklearn
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics

"""
An algorithm to classify hand written digits
"""
digits = load_digits()
data = scale(digits.data)  # We are scaling all of our features down to fit between the range of -1 to 1
y = digits.target

k = 10
samples, features = data.shape

#  https://scikit-learn.org/stable/modules/clustering.html#clustering-evaluation
def bench_k_means(estimator, name, data):  # Sklearn function to score our model
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y, estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')))


classifier = KMeans(n_clusters=k, init="random", n_init=10)
bench_k_means(classifier, "1", data)  # Training our data
