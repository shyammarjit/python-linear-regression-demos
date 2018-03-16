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
    #plt.pause(0.001)

#just plot x values in range 0, 10 step 0.5
x = np.arange(0, 10, 0.5)
print(x)
m = len(x)
#generate y values x^2-10x+25
y = np.power(x-5, 2)
print(y)

#convert to vectors

x_train = np.array([np.ones(len(x)), x, np.power(x, 2)])
print(x_train.shape)
y_train = (y[:, np.newaxis])

#h = theta0*x0 + theta1*x1 + theta2*x2 (x0=1 and x2=x^2)
theta = np.array(np.zeros((3, 1)))
x = 0
while True:
    x = x + 1
    h = theta.T.dot(x_train)
    error = h.T - y_train
    J = error.T.dot(error)/2*m;print(J)
    if(J<0.0001):
        break
    theta = theta - 0.001*x_train.dot(error)/m;
    update_line(g, x, J)
print(theta)
plt.show()
