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
    plt.pause(0.0001)

#just plot x values in range 0, 10 step 0.5
x = np.arange(0, 10, 0.5)
m = len(x)
#generate y values y=x+3
y = x + 3

#convert to vectors

x_train = np.array([np.ones(len(x)), x])
y_train = (y[:, np.newaxis])

#h = theta0*x0 + theta1*x1 + theta2*x2 (x0=1 and x2=x^2)
theta = np.array(np.zeros((2, 1)))
x = 0
xT = x_train.T
while True:
    J = 0
    #scan through training set
    for i in range(0, m):
        #for each training set
        x = x + 1;
        #calculate h_predicted
        h = xT[i].dot(theta)
        #calculate error=h_predicted-y_target
        error = (h - y_train[i])
        #accumulate error to J
        J = J + error*error
        #update theta for a training set
        theta = theta - 0.0001*(error*xT[i])[:, np.newaxis];
    J=J/m
    #plot J
    update_line(g, x, J)
    print(J)
    if(abs(J)<0.0001):
        break
print(theta)
plt.show()
