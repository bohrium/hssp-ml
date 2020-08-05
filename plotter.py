import matplotlib.pyplot as plt
import numpy as np

def plot_class(func):
    X, Y = np.meshgrid(
        np.linspace(-1.0,+1.0, 500),
        np.linspace(-1.0,+1.0, 500),
    )
    
    Z = func(X,-Y)
    Z = np.sign(Z)

    plt.imshow(Z, interpolation='nearest', extent=(-1,1,-1,1),
               cmap='coolwarm', vmin=-12.0, vmax=+12.0)

    plt.scatter([0],[0],color='black', s=10)

def r(deg=0):
    return 4.0**deg * np.random.randn()

data = [
    (max(-1.0, min(+1.0, round(r()/3.0,2))),
     max(-1.0, min(+1.0, round(r()/3.0,2))))
    for _ in range(36)
]


def plot_data(func): 
    styles = {
        +1:('red','+',90),
        -1:('blue','x',60)
    }
    for x,y in data:
        c,m,s = styles[+1 if func(x,y)>0.0 else -1]
        plt.scatter([x], [y], color=c, marker=m, s=s)

def render(file_name):
    plt.xticks([])
    plt.yticks([])
    plt.gca().axis('off')
    plt.gca().set_aspect('equal')
    plt.gcf().set_size_inches(5.0, 5.0)
    plt.tight_layout()
    plt.savefig(file_name)

todo = {
   'linr': lambda: ((lambda   b,c              : lambda X,Y:     b*X + c*Y)                                                                (     r(1),r(1)                                   )),
   'bias': lambda: ((lambda a,b,c              : lambda X,Y: a + b*X + c*Y)                                                                (r(0),r(1),r(1)                                   )),
   'quad': lambda: ((lambda a,b,c,d,e,f        : lambda X,Y: a + b*X + c*Y + d*X*X + e*X*Y + f*Y*Y)                                        (r(0),r(1),r(1),r(2),r(2),r(2)                    )),
   'cubc': lambda: ((lambda a,b,c,d,e,f,g,h,i,j: lambda X,Y: a + b*X + c*Y + d*X*X + e*X*Y + f*Y*Y + g*X*X*X + h*X*X*Y + i*X*Y*Y + j*Y*Y*Y)(r(0),r(1),r(1),r(2),r(2),r(2),r(3),r(3),r(3),r(3))),
}
for name, func_sampler in todo.items():
    for rep in range(10):
        func = func_sampler()
        plot_class(func)
        plot_data(func)
        render('db-{}-{}.png'.format(name, rep))
        plt.clf()
