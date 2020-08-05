import matplotlib.pyplot as plt
import numpy as np

##
##  Load Data
##
imgs = np.reshape(np.load('imgs.npy'), (-1, 28, 28))/256.0
lbls =            np.load('lbls.npy')               
print('loaded {} many {}-by-{} images'.format(*imgs.shape))
print('loaded {} many labels'.format(*lbls.shape))

shirts = (lbls==0)
jeans  = (lbls==1)
dresses= (lbls==3)
purses = (lbls==8)
boots  = (lbls==9)

intensities = np.mean(np.abs(imgs               ), axis=(1,2))
bottom_seps = np.mean(np.abs(imgs[:,14:,12:16]  ), axis=(1,2))
asymmetries = np.mean(np.abs(imgs-imgs[:,:,::-1]), axis=(1,2))
backgrounds = np.mean(np.abs(imgs < 0.01        ), axis=(1,2))

styles = {
    'red':  {'marker':'+', 'c':'red', 's':60},
    'blue': {'marker':'x', 'c':'blue','s':30},
}
def plot_data(N, clothes, x_axis, y_axis, text, color):
    X, Y = x_axis[clothes][:N], y_axis[clothes][:N]
    plt.scatter(X, Y, **styles[color])
    plt.text(np.mean(X)+0.1, np.mean(Y)-0.1, text, color=color,
             fontdict={'size':16, 'family':'serif'})

plot_data(100, dresses, intensities, bottom_seps, 'dresses', 'red')
plot_data(100, jeans  , intensities, bottom_seps, 'jeans', 'blue')

plt.xticks([])
plt.yticks([])
plt.xlabel('intensity')
plt.ylabel('bottom separation')
plt.gca().set_aspect('equal')
plt.gcf().set_size_inches(3.0, 5.0)
plt.tight_layout()

plt.savefig('yo.png')
