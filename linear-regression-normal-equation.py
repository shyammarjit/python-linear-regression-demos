import numpy as np
#for inverse matrix
from numpy.linalg import inv

x = np.arange(0, 10, 0.5) #input
y = x + 3 #output

#form matrix x and Y
X = np.array([np.ones(len(x)), x]).T
Y = (y[:, np.newaxis])

#apply normal equation
theta = inv(X.T.dot(X)).dot(X.T).dot(Y)

print(theta)#it return theta0=3 and theta1=1
