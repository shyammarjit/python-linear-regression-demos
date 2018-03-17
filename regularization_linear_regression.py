#we use polynominal h(x) = theta0*x0 + theta1*x1 + theta2*x2 + theta3*x3
import matplotlib.pyplot as plt
import numpy as np

plt.figure(1)

#just plot x values in range 0, 10 step 0.5
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

plt.scatter(x, y, marker=r'$\clubsuit$')
plt.hold(True)
#convert to vectors
x_train = np.array([np.ones(len(x)), x, np.power(x, 2), np.power(x, 3), np.power(x, 4), np.power(x, 5)])
y_train = (y[:, np.newaxis])

#h = theta0*x0 + theta1*x1 + theta2*x2 (x0=1 and x2=x^2)
theta = np.array(np.zeros((x_train.shape[0], 1)))

i = 0
alpha = 0.001
lamda = 10
preJ = 0
while (True):
    i = i + 1
    h = theta.T.dot(x_train)
    error = h.T - y_train
    J = (error.T.dot(error) + lamda*theta[1::,0].T.dot(theta[1::,0]))/2*m; print(J)
    if(preJ == 0):
        preJ = J
    if(preJ < J):
        break
    else:
        preJ = J
        
    tmp = alpha*x_train.dot(error)/m
    tmp2 = tmp[1::,0] + alpha*lamda*theta[1::,0]/m
    theta[0,0::] = theta[0,0::] - tmp[0,0::]
    theta[1::,0] = theta[1::,0] - tmp2

print(theta)
y_pred = theta.T.dot(x_train)
plt.plot(x, y_pred.T)
plt.show()
