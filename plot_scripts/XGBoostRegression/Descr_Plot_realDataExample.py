# This script illustrates the preprocessing of the real data sets and generates a descriptive representation.

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
import matplotlib.pyplot as plt


### load dataset - Boston Housing
data_set = 'housing.csv'
dataframe = pd.read_csv(fr'{url}/{data_set}', header=None)
# split into input (X) and output (y) variables
X, y = dataframe.iloc[:, :-1], dataframe.iloc[:, -1]
n_features = X.shape[1]

### get classes for y
# define grid
grid = [0, 10, 25, 45, 60]
grid_arr = np.array([grid, np.array(range(1,len(grid)+1)).tolist()]) # class_names = ['cheap','moderate','expensiv','luxurious',' ']

y_df = pd.DataFrame({'val': y}) # all data
y_df['trCL'] = (grid_arr[1][np.digitize(y_df.val, grid_arr[0])].astype(int)-1).astype(str)

### define different distribution for classes
# x range gamma
no_samples_gamma = 1000
x_gamma = np.linspace(0, max(grid), no_samples_gamma)
# get gamma distribution of y
param = stats.gamma.fit(y_df.val, floc=0)
pdf_fitted = stats.gamma.pdf(x_gamma, *param)
# define expertise as gamma distribution
y_gamma = stats.gamma.pdf(x_gamma, 3, 0, 7)
# random samples from distribution
rng = np.random.default_rng()
y_rs = rng.gamma(3,7, size=no_samples_gamma)
y_rs = y_rs[y_rs < 60][:len(y)] # reduce sample size to the same size as y and without values out of given x-Grid-range


### load dataset - ames housing
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/'

data_set = 'ames.csv'
ames = pd.read_csv(r'C:\Users\AdminRBG.SSt\source\repos\amsOS_PPC_FEST\Real_dummyData\AmesHousing.csv')
ya = ames.SalePrice.copy() / 1000.0
Xa = ames.drop('SalePrice', axis=1).copy()
# Remove columns with NaN or Inf values
Xa = Xa.drop(columns=['Lot Frontage', 'Garage Yr Blt', 'Mas Vnr Area'], axis=1)
Xa = pd.get_dummies(Xa)
Xa.columns = Xa.columns.astype(str)


### get classes for y
# define grid
grida = [0,100,250,500,800]
grida_arr = np.array([grida, np.array(range(1,len(grida)+1)).tolist()])

ya_df = pd.DataFrame({'val': ya}) # all data
ya_df['trCL'] = (grida_arr[1][np.digitize(ya_df.val, grida_arr[0])].astype(int)-1).astype(str)

### define different distribution for classes
# x range gamma
no_samples_gamma = 1000
xa_gamma = np.linspace(0, max(grida), no_samples_gamma)
# get gamma distribution of y
parama = stats.gamma.fit(ya_df.val, floc=0)
pdf_fitteda = stats.gamma.pdf(xa_gamma, *parama)
# define expertise as gamma distribution
ya_gamma = stats.gamma.pdf(xa_gamma, 6, 0, 40)
# random samples from distribution
rng = np.random.default_rng()
ya_rs = rng.gamma(6, 50, size=no_samples_gamma)
ya_rs = ya_rs[ya_rs < 800][:len(y)] # reduce sample size to the same size as y and without values out of given x-Grid-range


### descriptive stat. plot

# setting
font = {'size': 8}
matplotlib.rc('font', **font)
plt.rcParams.update({'font.family':'Times New Roman', 'text.color' : "black", 'axes.labelcolor' : "black"})

# plot
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)

ax1.hist(y, alpha = 0.4, density=True, label='Histogram (reg. target)')
for i in grid:
    ax1.axvline(i, linestyle='--', color='black', alpha=0.6)
ax1.plot(x_gamma, pdf_fitted, color='blue', label=f'PDF ~ Gamma({np.round(param[0],1)}, {np.round(param[2],1)})')
ax1.set_xlim(0,60); ax1.set_ylim(0,0.07)
ax1.set_ylabel('Boston Housing - Dataset')
ax1.legend(loc="upper right")

for i in grid:
    ax2.axvline(i, linestyle='--', color='black', alpha=0.6)
ax2.hist(y_rs, bins=grid, alpha=0.4, density=True, color='lightgreen', label='Histogram (clf target)')
ax2.plot(x_gamma, y_gamma, color='green', label='PDF ~ Gamma(3.0, 7.0)')
ax2.set_xlim(0,60); ax2.set_ylim(0,0.07)
ax2.legend(loc="upper right")

ax3.hist(ya, alpha = 0.4, density=True, label='Histogram (reg. target)')
for i in grida:
    ax3.axvline(i, linestyle='--', color='black', alpha=0.6)
ax3.plot(xa_gamma, pdf_fitteda, color='blue', label=f'PDF ~ Gamma({np.round(parama[0],1)}, {np.round(parama[2],1)})')
ax3.set_xlim(0,800); ax3.set_ylim(0,0.007)
ax3.set_ylabel('Ames - Dataset')
ax3.set_xlabel('$y^{reg}$ [in 1k \$]')
ax3.legend(loc="upper right")

for i in grida:
    ax4.axvline(i, linestyle='--', color='black', alpha=0.6)
ax4.hist(ya_rs, bins=grida, alpha=0.4, density=True, color='lightgreen', label='Histogram (clf target)')
ax4.plot(xa_gamma, ya_gamma, color='green', label='PDF ~ Gamma(6.0, 40.0)')
ax4.set_xlim(0,800); ax4.set_ylim(0,0.007)
ax4.set_xlabel('$y^{clf}$ (generated) [in 1k \$]')
ax4.legend(loc="upper right")

plt.tight_layout()
plt.show()
