import numpy
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from threading import Thread
import math
import time
import sys
import random

n = 2
if n<=10000:
    graph = numpy.ones((n,n)) - numpy.eye(n)

    graph /= graph.sum(axis=0)[:,numpy.newaxis]
    graph /= graph.sum(axis=1)[numpy.newaxis, :]
    graph /= n
else: 
    graph = None

data = numpy.random.uniform(0, 1, n)
S = data[:]
W = numpy.ones((n))
X = S/W

solution = data.mean()
alpha = 0.5



def main():
    try:
        for i in range(n*n*n):
            if graph is None:
                a = random.randint(0,n-1)
                b = random.randint(0,n-1)
                if a==b: continue
            else:
                a,b = divmod(numpy.random.choice(n*n, p=graph.flatten()), n)
            S[b] += alpha*S[a]
            W[b] += alpha*W[a]
            S[a] *= (1-alpha)
            W[a] *= (1-alpha)
            X[b] = S[b]/W[b]
            if i%n==0:
                time.sleep(0.3)
                print(i, f'{i/n}*n', f"log(n)={math.log(n)}")
    except KeyboardInterrupt:
        sys.exit(0)
Thread(target=main).start()


plot, = plt.plot(X, '*')
plt.plot([0,n],[solution, solution], "r-")
def animate(i):
    plot.set_ydata(X) 
    return plot,
animation.FuncAnimation(plt.gcf(), animate, range(200000), interval=25, blit=True)
plt.show()
