import matplotlib.pyplot as plt
import numpy as np

d = None
N = None

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

bottoms = sandals | boots | sneakers | trousers #| bags
tops    = tees | pullovers | shirts | coats | dresses

intensities = np.mean(np.abs(imgs               ), axis=(1,2))
diversities = np.mean(imgs*imgs, axis=(1,2)) - intensities**2
bottom_seps = np.mean(np.abs(imgs[:,14:,12:16]  ), axis=(1,2))
asymmetries = np.mean(np.abs(imgs-imgs[:,:,::-1]), axis=(1,2)) / (2*intensities)
heavinesses = np.mean(np.abs(imgs[:,:14,:]      ), axis=(1,2))
#backgrounds = np.mean(       imgs < 0.05         , axis=(1,2))
backgrounds = np.mean(       imgs < 0.01         , axis=(1,2))

styles = {
    'red':  {'marker':'+', 'c':'red', 's':60},
    'blue': {'marker':'x', 'c':'blue','s':30},
}
def plot_data(N, clothes, x_axis, y_axis, text, color):
    X, Y = x_axis[clothes][:N], y_axis[clothes][:N]
    plt.scatter(X, Y, **styles[color])
    plt.text(0.9-len(text)*0.02, np.mean(Y)+0.05, text, color=color, fontdict=fd)

def preprocess(x,y):
    x = 4*x-2
    y = 4*y-2
    #x = 3*x-1.5
    #y = 3*y-1.5
    return (
        1.0+0*x,
        y, x,
        x*x, x*y, y*y,
        x*x*x, x*x*y, x*y*y, y*y*y,
        x*x*x*x, x*x*x*y, x*x*y*y, x*y*y*y, y*y*y*y,
        x*x*x*x*x, x*x*x*x*y, x*x*x*y*y, x*x*y*y*y, x*y*y*y*y, y*y*y*y*y,
        x*x*x*x*x*x, x*x*x*x*x*y, x*x*x*x*y*y, x*x*x*y*y*y, x*x*y*y*y*y, x*y*y*y*y*y, y*y*y*y*y*y
    )[:d]

features = None 
test_features = None 

theta = np.zeros(len(preprocess(0,0)))
def classify(data):
    return np.dot(data, theta)

def badness(test=False):
    ff = test_features if test else features
    return (  np.sum(np.maximum(0.0, +1.0 - classify(ff['tops'])))
            + np.sum(np.maximum(0.0, +1.0 + classify(ff['bottoms'  ]))) )/sum(map(len, ff.values()))

def accuracy(test=False):
    ff = test_features if test else features
    return (  np.sum(np.abs(classify(ff['tops'])>0.0))
            + np.sum(np.abs(classify(ff['bottoms'])<0.0)) )/sum(map(len, ff.values()))

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
    Z = np.maximum(-3.0, np.minimum(3.0, Z))

    fadedness = 8.0 if db else 16.0
    plt.imshow(Z, interpolation='nearest', extent=(0,1,0,1),
               cmap='coolwarm', vmin=-fadedness, vmax=+fadedness)

def save_plot(file_name):
    plt.xticks([])
    plt.yticks([])
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])

    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)

    plt.xlabel('percent background', fontdict=fd)
    plt.ylabel('asymmetry', fontdict=fd)
    plt.gca().set_aspect('equal')
    plt.gcf().set_size_inches(5.0, 5.0)
    plt.tight_layout()
    
    plt.savefig(file_name)
    plt.clf()


for NN in [2, 16, 64]:
    N = NN

    dds = list(range(1,28+1)) if N==16 else [2,3,6,10,21,28]
    tests = []
    bads = []
    trains = []
    for dd in dds:
        d = dd

        features = {
            'tops':  np.array([ preprocess(x,y) for x,y in zip(backgrounds[tops  ][:N], asymmetries[tops  ][:N])]),
            'bottoms':np.array([preprocess(x,y) for x,y in zip(backgrounds[bottoms][:N], asymmetries[bottoms][:N])]),
        }
        TEST = 5000
        test_features = {
            'tops':  np.array([ preprocess(x,y) for x,y in zip(backgrounds[tops  ][-TEST:], asymmetries[tops  ][-TEST:])]),
            'bottoms':np.array([preprocess(x,y) for x,y in zip(backgrounds[bottoms][-TEST:], asymmetries[bottoms][-TEST:])]),
        }

        steps = []
        badnesses = []
        err_trains = []
        err_tests = []
        
        theta = np.zeros(len(preprocess(0,0)))

        step = 0
        while step != 10000:
            theta = theta + 0.01 * squeaky()
            step += 1
            
            if step % 25 and step not in [10, 1,2,4,8,16,32,64,128,256]: continue
        
            bb = badness()
            e_train = 1.0 - accuracy()
            e_test  = 1.0 - accuracy(test=True)
        
            steps.append(step)
            badnesses.append(bb)
            err_trains.append(e_train)
            err_tests.append(e_test)
        
            if step not in [1, 10, 100, 10000]: continue
            if (d,N) not in ((2,2), (2, 64), (10, 16), (28,2), (28, 64)): continue
        
            print('badness {:6.2f}% ... accuracy {:3.0f}% ... theta {:s}'.format(
                100*bb, 100*e_test, ' '.join('{:+2.0f}'.format(xx) for xx in theta)
            ))
        
            for db in [True, False]:
                plot_data(N, tops  ,  backgrounds, asymmetries, 'tops', 'red')
                plot_data(N, bottoms, backgrounds, asymmetries, 'bottoms', 'blue')
                plot_class(db)
                plt.text(0.02, 0.88, 'after {:3d} steps\ntest acc {:.0f}%'.format(
                    step, 100*(1.0-e_test)), fontdict=fd)
                save_plot('yo-{:02d}-{:04d}-{:s}-{:02d}.png'.format(d, N, ['soft','hard'][db], step))
        
        bads.append(badnesses[-1])
        trains.append(err_trains[-1])
        tests.append(err_tests[-1])

        if N != 16 or d not in (3,6,10,15,28): continue
        steps = np.array(steps)
        plt.plot(steps, err_tests , c='blue')
        plt.plot(steps, err_trains, c='red')
        plt.plot(steps, badnesses , c='orange')

        best_val, best_idx = min((v,i) for i,v in enumerate(err_tests))
        plt.plot([steps[best_idx], max(steps)     ], [best_val, best_val], c='blue', linestyle='dashed')
        plt.plot([steps[best_idx], steps[best_idx]], [0.0     , best_val], c='blue', linestyle='dashed')

        plt.xscale('log')
        plt.ylim([0.0, 0.1])
        plt.yticks([0.0, 0.1])
        plt.xlabel('(log) number of optimization updates', fontdict=fd)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)

        plt.tight_layout()
        plt.savefig('yoyo-{:02d}-{:04d}.png'.format(d,N))
        plt.clf()

    tests = np.array(tests)
    trains = np.array(trains)
    dds = np.array(dds)
    plt.plot(dds, trains, c='red')
    plt.plot(dds, tests, c='blue')
    plt.plot(dds, np.array(bads), c='orange')

    best_val, best_idx = min((v,i) for i,v in enumerate(tests))
    plt.plot([dds[best_idx], max(dds)],      [best_val, best_val], c='blue', linestyle='dashed')
    plt.plot([dds[best_idx], dds[best_idx]], [0.0, best_val],      c='blue', linestyle='dashed')

    plt.ylim(([0.0, 0.1]))
    plt.yticks(([0.0, 0.1]))
    plt.xticks(([1,3,6,10,15,21,28]))
    plt.xlabel('data dimension after preprocessing', fontdict=fd)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    plt.tight_layout()
    plt.savefig('yoyoyo-{:04d}.png'.format(N))
    plt.clf()
