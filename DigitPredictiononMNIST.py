import itertools

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.datasets import fetch_openml

from sklearn.metrics import accuracy_score

# Load data from website
X, y = fetch_openml('mnist_784', version = 1, return_X_y = True)
X = X/255
# rescale the data, use the traditional train/test split
  # (60K: Train) and (10K: Test)
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

# 2.a
def evaluateSVM(svm):
    svClassifier = svm.fit(X,y)
    yPred = svClassifier.predict(X_test)
    return yPred

# 1
# kernel = linear 
# C = 10
svClassifier = SVC(kernel = 'linear', C = 10)
yPred = evaluateSVM(svClassifier)
print(1-accuracy_score(y_test,yPred))

# 2
# kernel = linear
# gamma = scale
# C = 0.25
# tol = 1e-4
svClassifier = SVC(kernel = 'linear', C = 0.25, gamma = 'scale', tol = 1e-4)
yPred = evaluateSVM(svClassifier)
print(1-accuracy_score(y_test,yPred))

# 3
# kernel = poly 
# max_iter = 50
# degree = 5
svClassifier = SVC(kernel = 'poly', max_iter = 50, degree = 3)
yPred = evaluateSVM(svClassifier)
print(1-accuracy_score(y_test,yPred))

# 4
# kernel = poly
# degree = 10
# max_iter = 100
svClassifier = SVC(kernel = 'poly', max_iter = 500, degree = 10)
yPred = evaluateSVM(svClassifier)
print(1-accuracy_score(y_test,yPred))

# 5
# kernel = sigmoid 
# C = 1
# tol = 0.1
svClassifier = SVC(kernel = 'sigmoid', C = 1, tol = 0.1)
yPred = evaluateSVM(svClassifier)
print(1-accuracy_score(y_test,yPred))

# 6
# kernel = rbf 
# gamma = scale
# class_weight = balanced
svClassifier = SVC(kernel = 'rbf', gamma = 'scale', class_weight = 'balanced')
yPred = evaluateSVM(svClassifier)
print(1-accuracy_score(y_test,yPred))

# 7
# kernel = sigmoid
# coef0 = 0.8
svClassifier = SVC(kernel = 'sigmoid', coef0 = 0.8)
yPred = evaluateSVM(svClassifier)
print(1-accuracy_score(y_test,yPred))

# 8
# kernel = sigmoid
# coef0 = 50
# max_iter = 100
# class_weight = balanced
svClassifier = SVC(kernel = 'sigmoid', coef0 = 50, max_iter = 100, class_weight = 'balanced')
yPred = evaluateSVM(svClassifier)
print(1-accuracy_score(y_test,yPred))

# 9
# Kernel = rbf 
# tol = 100
# gamma = auto
# degree = 10
svClassifier = SVC(kernel = 'rbf', tol = 100, gamma = 'auto', degree = 10)
yPred = evaluateSVM(svClassifier)
print(1-accuracy_score(y_test,yPred))

# 10
# Kernel = rbf 
# tol = 0.001
# gamma = auto
# degree = 10
svClassifier = SVC(kernel = 'rbf', tol = 0.001, gamma = 'auto', degree = 10)
yPred = evaluateSVM(svClassifier)
print(1-accuracy_score(y_test,yPred))



## 2.b MLP
def evaluateMLP(mlpClassifier):
    mlp = mlpClassifier.fit(X_train, y_train)
    predicted_y = mlp.predict(X_test)
    return predicted_y

# 1
# solver: sgd
# alpha: default
# learning_rate: invscaling
# power_t: default
# shuffle: true
mlpClassifier = MLPClassifier(solver = "sgd", learning_rate = "invscaling", shuffle = True, max_iter = 500)
yPred = evaluateMLP(mlpClassifier)
print(1 - accuracy_score(y_test,yPred))

# 2
# learning_rate_init: 0.1
# aplha: 0.001
# max_iter: 100
mlpClassifier = MLPClassifier(learning_rate_init = 0.01, alpha = 0.001, max_iter = 100 )
yPred = evaluateMLP(mlpClassifier)
print(1 - accuracy_score(y_test,yPred))

# 3
# solver: sgd
# alpha: 1
# learning_rate_init: 0.001
# learning_rate: constantmomentum: 0.01

mlpClassifier = MLPClassifier(learning_rate_init = 0.001, alpha = 1, learning_rate = "constant",solver = "sgd")
yPred = evaluateMLP(mlpClassifier)
print(1 - accuracy_score(y_test,yPred))

# 4
# hidden_layer_sizes : [10,10]
# activation: tanh
# alpha: 0.1
# max_iter: 500

mlpClassifier = MLPClassifier(hidden_layer_sizes = [10,10], activation = "tanh", alpha = 0.1, max_iter = 500)
yPred = evaluateMLP(mlpClassifier)
print(1 - accuracy_score(y_test,yPred))

# 5
# solver: lbfgs
# max_iter: 1000
# early_stopping: true
# activation: identity
# learning_rate_init: default 

mlpClassifier = MLPClassifier(solver = "lbfgs", activation = "identity", early_stopping = True, max_iter = 1000)
yPred = evaluateMLP(mlpClassifier)
print(1 - accuracy_score(y_test,yPred))


# 6
# solver: lbfgs
# hidden_layer_sizes: [500,500] 
# alpha: 10
# early_stopping: true
# max_iter: 2000
# activation: default

mlpClassifier = MLPClassifier(solver = "lbfgs", alpha = 10, early_stopping = True, max_iter = 2000, hidden_layer_sizes = [300,300])
yPred = evaluateMLP(mlpClassifier)
print(1 - accuracy_score(y_test,yPred))

# 7
#hidden_layer_sizes : [10,10]
#activation: tanh
#alpha: 0.1
#max_iter: 500

mlpClassifier = MLPClassifier(alpha = 0.2, max_iter = 500, hidden_layer_sizes = [10,10], activation = "tanh")
yPred = evaluateMLP(mlpClassifier)
print(1 - accuracy_score(y_test,yPred))

# 8
# solver: adam
# activation: ‘relu’
# alpha: 1
# hidden_layer_sizes = (150,100,50,50)
# random_state = 1
# max_iter = 750
mlpClassifier = MLPClassifier(solver = 'adam', activation = 'relu', alpha = 1, max_iter = 750, hidden_layer_sizes = [150,100,50,50],  random_state = 1)
yPred = evaluateMLP(mlpClassifier)
print(1 - accuracy_score(y_test,yPred))

# 9
#learning_rate_init = 0.0001
#alpha = 10

mlpClassifier = MLPClassifier(learning_rate_init = 0.0001, alpha = 10)
yPred = evaluateMLP(mlpClassifier)
print(1 - accuracy_score(y_test,yPred))

# 10
# learning_rate_init = 10
# alpha = 0.001
mlpClassifier = MLPClassifier(learning_rate_init = 10, alpha = 0.001)
yPred = evaluateMLP(mlpClassifier)
print(1 - accuracy_score(y_test,yPred))

# 11
# learning_rate_init: 0.0001
# alpha: 0.001
# max_iter: 500
mlpClassifier = MLPClassifier(learning_rate_init = 0.0001, alpha = 0.001, max_iter = 500)
yPred = evaluateMLP(mlpClassifier)
print(1 - accuracy_score(y_test,yPred))

# 12
#learning_rate: 10
#alpha: 10
mlpClassifier = MLPClassifier(learning_rate_init = 10, alpha = 10)
yPred = evaluateMLP(mlpClassifier)
print(1 - accuracy_score(y_test,yPred))


# 2.c
## KNN
def evaluateKNN(knnClassifier):
  knn = knnClassifier.fit(X_train, y_train)
  predicted_y = knn.predict(X_test)
  return predicted_y

#1
#n_neighbors:  10
#weights:  distance 
#p:  2
model = KNeighborsClassifier(n_neighbors = 10, p = 2, weights="distance")
yPred = evaluateKNN(model)
error = (1-accuracy_score(y_test,yPred)) 
print(error)

#2
#n_neighbors:  2 
#weights:  distance 
#p: 4
#algorithm:  kd_tree
model = KNeighborsClassifier(n_neighbors = 2, p = 4, algorithm = "kd_tree")
yPred = evaluateKNN(model)
error = (1-accuracy_score(y_test,yPred)) 
print(error)

#3
#n_neighbors:  3 
#weights:  distance 
model = KNeighborsClassifier(n_neighbors = 3, weights="distance")
yPred = evaluateKNN(model)
error = (1-accuracy_score(y_test,yPred)) 
print(error)

#4
#n_neighbors:  6 
#leaf_size:  10
#algorithm: Distance
model = KNeighborsClassifier(n_neighbors = 6, leaf_size = 10, weights="distance")
yPred = evaluateKNN(model)
error = (1-accuracy_score(y_test,yPred)) 
print(error)

# 5
#n_neighbors:  7
#weights: Uniform
#leaf_size:: 50
#algorithm: ball tree
model = KNeighborsClassifier(n_neighbors = 7, weights="uniform", leaf_size = 50, algorithm = "ball_tree")
yPred = evaluateKNN(model)
error = (1-accuracy_score(y_test,yPred)) 
print(error)

# 6
#n_neighbors:  9
#weights: Uniform
#leaf_size: 100
#algorithm: auto
model = KNeighborsClassifier(n_neighbors = 9, weights="uniform", leaf_size = 100, algorithm = "auto")
yPred = evaluateKNN(model)
error = (1-accuracy_score(y_test,yPred)) 
print(error)

#7
#n_neighbors:  1
#algorithm: auto
model = KNeighborsClassifier(n_neighbors = 1, algorithm = "auto")
yPred = evaluateKNN(model)
error = (1-accuracy_score(y_test,yPred)) 
print(error)

#8
#n_neighbors:  2 
#weights:  uniform 
#algorithm:  brute 
model = KNeighborsClassifier(n_neighbors = 2, weights="uniform", algorithm = "brute")
yPred = evaluateKNN(model)
error = (1-accuracy_score(y_test,yPred)) 
print(error)

#9
#n_neighbors:  7 
#Weights:  distance
#n_jobs = 10
model = KNeighborsClassifier(n_neighbors = 7, weights="distance", n_jobs = 10)
yPred = evaluateKNN(model)
error = (1-accuracy_score(y_test,yPred)) 
print(error)

#10
#n_jobs = 10
#n_neighbors:  15
#algorithm:  brute
#Weights:  distance
model = KNeighborsClassifier(n_neighbors = 15, weights="distance", algorithm = "brute", n_jobs = 10)
yPred = evaluateKNN(model)
error = (1-accuracy_score(y_test,yPred)) 
print(error)