import numpy as np

# Import the data from a .mat file using scipy
from scipy.io import loadmat
x = loadmat('JM_tauth_data.mat')
a0 = x['a0']
b0 = x['b0']
delta = x['delta']
ip = x['ip']
kappa = x['kappa']
nebar = x['nebar']
ploss = x['ploss']
r0 = x['r0']
tauth = x['tauth']
zeff = x['zeff']

# Concatenate all the inputs into one input matrix
X = np.concatenate((a0,b0,delta,ip,kappa,nebar,ploss,r0,zeff),axis=1)

# Assign the output to tauth array 
y = tauth

## Split between train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.47, random_state = 100)

#Import Mean square error to try to compare the methods
from sklearn.metrics import mean_squared_error



### Multi Layer Perceptron classifier
#####################################
print("Using a Mutli Layer Perceptron...")

from sklearn.neural_network import MLPRegressor

# Fit the Model
MLP = MLPRegressor(solver='lbfgs', random_state=100, max_iter=100)
MLP.fit(x_train, y_train)

#Predict the values
y_predicted= MLP.predict(x_test)

# Display the error
error = mean_squared_error(y_test, y_predicted)
print ("Error of the Multi Layer Perceptron is: ",error)




###    Support Vector Machine
#############################
print("Using a Support Vector Machine...")
from sklearn import svm
from sklearn.model_selection import cross_val_score

# Fit the Model
SVM = svm.SVR()
SVM.fit(x_train, y_train) 

#Predict the values
y_predicted = SVM.predict(x_test)

# Display the error
error = mean_squared_error(y_test, y_predicted)
print ("Error of the Support Vector Machine is: ",error)

scores =cross_val_score(SVM, x_train, y_train, cv=6)
print(scores)
print(scores.mean())




###    Gaussian Process
#######################
print("Using Gaussian Process...")
from sklearn.gaussian_process import GaussianProcessRegressor

# Fit the Model
GPR = GaussianProcessRegressor()
GPR.fit(x_train, y_train) 

#Predict the values
y_predicted = GPR.predict(x_test)

# Display the error
error = mean_squared_error(y_test, y_predicted)
print ("Error of the Gaussian Process is: ",error)
