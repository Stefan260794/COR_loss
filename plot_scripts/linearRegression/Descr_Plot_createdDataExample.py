# This script illustrates the creation of the examined data and generates a descriptive representation.

# packages
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets
import scipy.stats as stats

### Create Data set

# simulate dummy dataset
X, y = datasets.make_regression(n_samples=1000, n_features=1, noise=20, random_state=4)

# define a grid for classification optimization
grid = [-200,-50,50,150,200] # grid
grid = [-300,-200,-50,50,150,200,300] # extendet grid necessary for training process
grid_arr = np.array([grid, np.array(range(1,len(grid)+1)).tolist()])

# get class for each value of y
y_df = pd.DataFrame({'val': y})
y_df['trCL'] = (grid_arr[1][np.digitize(y_df.val, grid_arr[0])].astype(int)-1).astype(str)

### find gamma distribution of y
# x range gamma
no_samples_norm = 1100
x_norm = np.linspace(min(grid), max(grid), no_samples_norm)
# get gamma distribution of y
param = stats.norm.fit(y_df.val, floc=0)
pdf_fitted = stats.norm.pdf(x_norm, *param)

# define expertise as new gamma distribution (with random selected parameter)
mu = 5; sigma=65
y_norm = stats.norm.pdf(x_norm, mu, sigma)

# random samples from distribution
rng = np.random.default_rng()
y_rs = rng.normal(mu, sigma, size=no_samples_norm)

df = pd.concat([pd.DataFrame(X), y_df], axis=1) # split data together to stratify with trCL and genCL
df = df.sort_values(by='val')

geny = y_rs[:len(y)] # reduce sample size to the same size as y and without values out of given x-Grid-range
geny = np.sort(geny)

# add gen data to dataframe
df['geny'] = geny
df['genCL'] = (grid_arr[1][np.digitize(df.geny, grid_arr[0])].astype(int)-1).astype(str)

### descriptive stat. plot
# settings
font = {'size': 8}
matplotlib.rc('font', **font)
plt.rcParams.update({'font.family':'Times New Roman', 'text.color' : "black", 'axes.labelcolor' : "black"})

# descriptive plot
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
plt.show()


