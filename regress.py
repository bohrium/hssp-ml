import matplotlib.pyplot as plt
import numpy as np

fd={'size':16, 'family':'serif'}

##
##  Load Data
##
imgs = np.reshape(np.load('imgs.npy'), (-1, 28, 28))/256.0
lbls =            np.load('lbls.npy')               
print('loaded {} many {}-by-{} images'.format(*imgs.shape))
print('loaded {} many labels'.format(*lbls.shape))

tees     = (lbls==0)
trousers = (lbls==1)
pullovers= (lbls==2)
dresses  = (lbls==3)
coats    = (lbls==4)
sandals  = (lbls==5)
shirts   = (lbls==6)
sneakers = (lbls==7)
bags     = (lbls==8)
boots    = (lbls==9)

bottoms = sandals | boots | sneakers | trousers
tops    = tees | pullovers | shirts | coats

intensities = np.mean(np.abs(imgs               ), axis=(1,2))
diversities = np.mean(imgs*imgs, axis=(1,2)) - intensities**2
bottom_seps = np.mean(np.abs(imgs[:,14:,12:16]  ), axis=(1,2))
asymmetries = np.mean(np.abs(imgs-imgs[:,:,::-1]), axis=(1,2)) / intensities
heavinesses = np.mean(np.abs(imgs[:,:14,:]      ), axis=(1,2))
backgrounds = np.mean(       imgs < 0.05         , axis=(1,2))

styles = {
    'red':  {'marker':'+', 'c':'red', 's':60},
    'blue': {'marker':'x', 'c':'blue','s':30},
}
def plot_data(N, clothes, x_axis, y_axis, text, color):
    X, Y = x_axis[clothes][:N], y_axis[clothes][:N]
    plt.scatter(X, Y, **styles[color])
    plt.text(np.mean(X)+0.1, np.mean(Y)-0.1, text, color=color, fontdict=fd)

def preprocess(x,y):
    return (
        1.0+0*x, x, y, x*x, x*y, y*y, x*x*x, x*x*y, x*y*y, y*y*y, x*x*x*x, x*x*x*y, x*x*y*y, x*y*y*y, y*y*y*y
        #1.0+0*x, x, y, x*x, x*y, y*y, x*x*x, x*x*y, x*y*y, y*y*y
        #1.0+0*x, x, y
    )

N = 50
features = {
    'tops':  np.array([ preprocess(x,y) for x,y in zip(backgrounds[tops  ][:N], asymmetries[tops  ][:N])]),
    'bottoms':np.array([preprocess(x,y) for x,y in zip(backgrounds[bottoms][:N], asymmetries[bottoms][:N])]),
}

theta = np.zeros(len(preprocess(0,0)))
def classify(data):
    return np.dot(data, theta)

def badness():
    return (  np.sum(np.maximum(0.0, +1.0 - classify(features['tops'])))
            + np.sum(np.maximum(0.0, +1.0 + classify(features['bottoms'  ]))) )/(2*N)

def accuracy():
    return (  np.sum(np.abs(classify(features['tops'])>0.0))
            + np.sum(np.abs(classify(features['bottoms'])<0.0)) )/(2*N)

def squeaky():
    return (  np.sum(features['tops'][classify(features['tops']) < +1.0], axis=0)
            - np.sum(features['bottoms'][classify(features['bottoms']) > -1.0], axis=0) )/(2*N)

def plot_class(db=True):
    X, Y = np.meshgrid(
        np.linspace(0.0,+1.0, 500),
        np.linspace(0.0,+1.0, 500),
    )
    Y = 1.0-Y
    Z = np.stack(preprocess(X,Y), axis=2)
    Z = classify(Z)
    if db:
        Z = np.sign(Z)

    plt.imshow(Z, interpolation='nearest', extent=(0,1,0,1),
               cmap='coolwarm', vmin=-12.0, vmax=+12.0)

def save_plot(file_name):
    plt.xticks([])
    plt.yticks([])
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])

    plt.xlabel('backgrounds', fontdict=fd)
    plt.ylabel('asymmetries', fontdict=fd)
    plt.gca().set_aspect('equal')
    plt.gcf().set_size_inches(8.0, 8.0)
    plt.tight_layout()
    
    plt.savefig(file_name)
    plt.clf()

steps_taken = 0
for i in range(20):
    nb_steps = 20*(i+1)
    for _ in range(nb_steps):
        theta = theta + 0.1 * squeaky()
    steps_taken += nb_steps 

    print('badness {:6.2f}% ... accuracy {:3.0f}%'.format(
          100*badness(), 100*accuracy()), end='   ')
    print(' '.join('{:+2.0f}'.format(xx) for xx in theta))

    for db in [True, False]:
        plot_data(N, tops  ,   backgrounds, asymmetries, 'tops', 'red')
        plot_data(N, bottoms, backgrounds, asymmetries, 'bottoms', 'blue')
        plot_class(db)
        plt.text(0.02, 0.95, 'after {:3d} steps'.format(steps_taken), fontdict=fd)
        save_plot('yo-{:02d}-{:s}.png'.format(i, ['sm','bd'][db]))

