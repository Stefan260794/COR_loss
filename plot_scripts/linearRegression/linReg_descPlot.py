#%%
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import numpy as np
import pandas as pd
import scipy.stats as stats



class LinearRegression:

    def __init__(self, lr = 0.001, n_iters=1000, alpha = 1):
        self.lr = lr
        self.n_iters = n_iters
        self.alpha = alpha
        self.weights = None
        self.bias = None

    def fit(self, X, y):

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias

            # mse loss
            dw = (1/n_samples) * np.dot(X.T, (y_pred-y))
            db = (1/n_samples) * np.sum(y_pred-y)

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db
    
    def fit_COR(self, X, y):

        y_true = yreg = y[:,0]
        yclf = y[:,1]

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias

            # COR loss
            # 1. get class of prediciton
            temp = pd.DataFrame({'pred': y_pred})
            temp['predCL'] = (grid_arr[1][np.digitize(y_pred, grid_arr[0])].astype(int)-1).astype(int)
            # 2. get true class
            temp['trueCL'] = yclf.astype(int)
            # 2.2 abstnad klasse
            temp['deltaCL'] = temp.trueCL - temp.predCL
            # 3. compare classes
            temp['grad2'] = 0
            temp.loc[temp.deltaCL > 0,'grad2'] = -(grid_arr[0][temp[temp.deltaCL > 0]['trueCL']-1] + 1e-3 - temp[temp.deltaCL > 0].pred)
            temp.loc[temp.deltaCL < 0,'grad2'] = -(grid_arr[0][temp[temp.deltaCL < 0]['trueCL']] - 1e-3 - temp[temp.deltaCL < 0].pred)
            # final loss
            grad = self.alpha * (y_pred - y_true) + (1-self.alpha) * temp.grad2

            dw = (1/n_samples) * np.dot(X.T, grad)
            db = (1/n_samples) * np.sum(grad)

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred


# %%
# dummy dataset
X, y = datasets.make_regression(n_samples=1000, n_features=1, noise=20, random_state=4)

#% find gamma dist
grid = [-200,-50,50,150,200]
grid = [-300,-200,-50,50,150,200,300]
grid_arr = np.array([grid, np.array(range(1,len(grid)+1)).tolist()]) # , class_names])

y_df = pd.DataFrame({'val': y}) # all data
y_df['trCL'] = (grid_arr[1][np.digitize(y_df.val, grid_arr[0])].astype(int)-1).astype(str)

# -------------
# x range gamma
no_samples_norm = 1100
x_norm = np.linspace(min(grid), max(grid), no_samples_norm)
# get gamma dist of y
param = stats.norm.fit(y_df.val, floc=0)
param
pdf_fitted = stats.norm.pdf(x_norm, *param)

# define expertise as gamma dist
mu = 5; sigma=65
y_norm = stats.norm.pdf(x_norm, mu, sigma)
# random samples from distribution
rng = np.random.default_rng()
y_rs = rng.normal(mu, sigma, size=no_samples_norm)

# add gen-data to true data
df = pd.concat([pd.DataFrame(X), y_df], axis=1) # split data together to stratify with trCL and genCL
df = df.sort_values(by='val')

geny = y_rs[:len(y)] # reduce sample size to the same size as y and without values out of given x-Grid-range
geny = np.sort(geny)

# add gen data to dataframe
df['geny'] = geny
df['genCL'] = (grid_arr[1][np.digitize(df.geny, grid_arr[0])].astype(int)-1).astype(str)

#%% descriptive stat plot
font = {'size': 8}
matplotlib.rc('font', **font)
plt.rcParams.update({'font.family':'Times New Roman', 'text.color' : "black", 'axes.labelcolor' : "black"})


# 1. plot true class dist
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(7,3.5))

ax1.hist(df.val, alpha = 0.4, density=True, label='Histogram (reg. target)') # true values
for i in grid:
    ax1.axvline(i, linestyle='--', color='black', alpha=0.6)
ax1.plot(x_norm, pdf_fitted, color='blue', label=f'PDF ~ Normal({np.round(param[0],1)}, {np.round(param[1],1)})') # fitted gamma dist on true values
ax1.set_xlim(-210,210); ax1.set_ylim(0,0.014)
ax1.set_xlabel('$y^{reg}$')
ax1.set_ylabel('Generated Data')
ax1.legend(loc="upper right")

for i in grid:
    ax2.axvline(i, linestyle='--', color='black', alpha=0.6) # grid
ax2.hist(y_rs, bins=grid, alpha=0.4, density=True, color='lightgreen', label='Histogram (clf target)')
ax2.plot(x_norm, y_norm, color='green', label=f'PDF ~ Normal({mu}, {sigma})') # verteilung
ax2.set_xlabel('$y^{clf}$')
ax2.set_xlim(-210,210); ax2.set_ylim(0,0.014)
ax2.legend(loc="upper right")

plt.tight_layout()
# plt.show()

plt.savefig(r'C:\Users\AdminRBG.SSt\Desktop\2Paper_result_neuAusfuehrung\linReg\plots\plotdescrLinReg_example.png', dpi=500)

