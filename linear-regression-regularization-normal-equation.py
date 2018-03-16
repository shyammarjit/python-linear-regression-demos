import numpy as np
#for inverse matrix
from numpy.linalg import pinv
import matplotlib.pyplot as plt

x = np.array([-0.99768,
 -0.69574,
 -0.40373,
 -0.10236,
 0.22024,
 0.47742,
 0.82229])
m = len(x)
#generate y values x^2-10x+25
y = np.array([2.0885,
 1.1646,
 0.3287,
 0.46013,
 0.44808,
 0.10013,
 -0.32952])


#form matrix x and Y
X = np.array([np.ones(len(x)), x, np.power(x, 2), np.power(x, 3), np.power(x, 4), np.power(x, 5)]).T
Y = (y[:, np.newaxis])
one = np.identity(X.shape[1])
one[0,0] = 0

lamda = 2
#apply normal equation
theta = pinv(X.T.dot(X) + lamda*one).dot(X.T).dot(Y)

print(theta)

y_pred = theta.T.dot(X.T)

plt.scatter(x, y,  color='black')
plt.plot(x, y_pred[0,:], color='blue', linewidth=3)

plt.show()
