import matplotlib.pyplot as plt
import numpy as np

plt.figure(1)
ax = plt.gca()
ax.set_autoscale_on(True)
plt.ylabel('error')
plt.xlabel('step')
px = [0]
py = [0]
g, = plt.plot(px, py)

def update_line(g, x, y):
    global px
    global py
    if(len(px) == 60):
        px = []
        py = []
    if(len(px) == 30):
        plt.text(x, y, str(y))
    px = np.append(px, x)
    py = np.append(py, y)
    g.set_xdata(px)
    g.set_ydata(py)
    ax.relim()
    ax.autoscale_view(True,True,True)
    plt.draw()
    plt.pause(0.001)

#generate training set
x = np.arange(0, 10, 0.5) #input
m = len(x)
y = x + 3 #output

#convert to vectors
x_train = np.array([np.ones(len(x)), x])
y_train = (y[:, np.newaxis])

#h = theta0*x0 + theta1*x1 (x0=1)
theta = np.array(np.zeros((2, 1)))

#iterator 500 steps
for x in range(0, 500):
    h = theta.T.dot(x_train) #h=thetaT*x
    error = h.T - y_train #error=h_predict-y_target
    J = error.T.dot(error)/2*m; #cost function J
    #update theta using batch gradient descent    
    theta = theta - 0.06*x_train.dot(error)/m;
    #plot J
    update_line(g, x, J)

#finsih training and print theta
print(theta)
plt.show()
