#
# https://machinelearningmastery.com/results-for-standard-classification-and-regression-machine-learning-datasets/

# mit early stopping and CV and hypertuning
# split train, val und test
# normalized

# %%
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
import matplotlib.pyplot as plt

font = {'size': 8}
matplotlib.rc('font', **font)
plt.rcParams.update({'font.family':'Times New Roman', 'text.color' : "black", 'axes.labelcolor' : "black"})


# %%
# regression mlp model for the abalone dataset
# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/'

# data_set = 'ames.csv'
data_set = 'housing.csv'
dataframe = pd.read_csv(fr'{url}/{data_set}', header=None)
# split into input (X) and output (y) variables
X, y = dataframe.iloc[:, :-1], dataframe.iloc[:, -1]
n_features = X.shape[1]


# define unbiased classes
grid = [0, 10, 25, 45, 60] # grid for housing
class_names = ['cheap','moderate','expensiv','luxurious',' ']
grid_arr = np.array([grid, np.array(range(1,len(grid)+1)).tolist()]) # , class_names])

y_df = pd.DataFrame({'val': y}) # all data
y_df['trCL'] = (grid_arr[1][np.digitize(y_df.val, grid_arr[0])].astype(int)-1).astype(str)

# -------------
# x range gamma
no_samples_gamma = 1000
x_gamma = np.linspace(0, max(grid), no_samples_gamma)
# get gamma dist of y
param = stats.gamma.fit(y_df.val, floc=0)
pdf_fitted = stats.gamma.pdf(x_gamma, *param)
# define expertise as gamma dist
y_gamma = stats.gamma.pdf(x_gamma, 3, 0, 7)
# random samples from distribution
rng = np.random.default_rng()
y_rs = rng.gamma(3,7, size=no_samples_gamma)
y_rs = y_rs[y_rs < 60][:len(y)] # reduce sample size to the same size as y and without values out of given x-Grid-range



# --------------
# ames housing


url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/'
data_set = 'ames.csv'
ames = pd.read_csv(r'C:\Users\AdminRBG.SSt\source\repos\amsOS_PPC_FEST\Real_dummyData\AmesHousing.csv')
ya = ames.SalePrice.copy() / 1000.0
Xa = ames.drop('SalePrice', axis=1).copy()
# Remove columns with NaN or Inf values
Xa = Xa.drop(columns=['Lot Frontage', 'Garage Yr Blt', 'Mas Vnr Area'], axis=1)
Xa = pd.get_dummies(Xa)
Xa.columns = Xa.columns.astype(str)


# define unbiased classes
grida = [0,100,250,500,800] # grid for housing
class_names = ['cheap','moderate','expensiv','luxurious',' ']
grida_arr = np.array([grida, np.array(range(1,len(grida)+1)).tolist()]) # , class_names])

ya_df = pd.DataFrame({'val': ya}) # all data
ya_df['trCL'] = (grida_arr[1][np.digitize(ya_df.val, grida_arr[0])].astype(int)-1).astype(str)

# -------------
# x range gamma
no_samples_gamma = 1000
xa_gamma = np.linspace(0, max(grida), no_samples_gamma)
# get gamma dist of y
parama = stats.gamma.fit(ya_df.val, floc=0)
pdf_fitteda = stats.gamma.pdf(xa_gamma, *parama)
# define expertise as gamma dist
ya_gamma = stats.gamma.pdf(xa_gamma, 6, 0, 40)
# random samples from distribution
rng = np.random.default_rng()
ya_rs = rng.gamma(6, 50, size=no_samples_gamma)
ya_rs = ya_rs[ya_rs < 800][:len(y)] # reduce sample size to the same size as y and without values out of given x-Grid-range



# ------------------------------------------------

# 1. plot true class dist
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)

ax1.hist(y, alpha = 0.4, density=True, label='Histogram (reg. target)') # true values
for i in grid:
    ax1.axvline(i, linestyle='--', color='black', alpha=0.6)
# ax1.hist(y_df.val, bins=grid, alpha=0.4, density=True) # given grid with true values
ax1.plot(x_gamma, pdf_fitted, color='blue', label=f'PDF ~ Gamma({np.round(param[0],1)}, {np.round(param[2],1)})') # fitted gamma dist on true values
ax1.set_xlim(0,60); ax1.set_ylim(0,0.07)
# ax1.set_title(f'y ~ Gamma({np.round(param[0],1)}, {np.round(param[2],1)})', fontsize=8)
ax1.set_ylabel('Boston Housing - Dataset')
ax1.legend(loc="upper right")

# ax2.hist(y_rs, alpha = 0.4, density=True, color='green') # shifted values
for i in grid:
    ax2.axvline(i, linestyle='--', color='black', alpha=0.6) # grid
ax2.hist(y_rs, bins=grid, alpha=0.4, density=True, color='lightgreen', label='Histogram (clf target)')
ax2.plot(x_gamma, y_gamma, color='green', label='PDF ~ Gamma(3.0, 7.0)') # verteilung
ax2.set_xlim(0,60); ax2.set_ylim(0,0.07)
ax2.legend(loc="upper right")

ax3.hist(ya, alpha = 0.4, density=True, label='Histogram (reg. target)') # true values
for i in grida:
    ax3.axvline(i, linestyle='--', color='black', alpha=0.6)
# ax3.hist(ya_df.val, bins=grida, alpha=0.4, density=True) # given grid with true values
ax3.plot(xa_gamma, pdf_fitteda, color='blue', label=f'PDF ~ Gamma({np.round(parama[0],1)}, {np.round(parama[2],1)})') # fitted gamma dist on true values
ax3.set_xlim(0,800); ax3.set_ylim(0,0.007)
ax3.set_ylabel('Ames - Dataset')
ax3.set_xlabel('$y^{reg}$ [in 1k \$]')
ax3.legend(loc="upper right")

# ax4.hist(ya_rs, alpha = 0.4, density=True, color='green') # shifted values
for i in grida:
    ax4.axvline(i, linestyle='--', color='black', alpha=0.6) # grid
ax4.hist(ya_rs, bins=grida, alpha=0.4, density=True, color='lightgreen', label='Histogram (clf target)')
ax4.plot(xa_gamma, ya_gamma, color='green', label='PDF ~ Gamma(6.0, 40.0)') # verteilung
ax4.set_xlim(0,800); ax4.set_ylim(0,0.007)
ax4.set_xlabel(r'$y^{clf}$ (generated) [in 1k \$]')
ax4.legend(loc="upper right")

plt.tight_layout()
# plt.show()

plt.savefig(r'C:\Users\AdminRBG.SSt\Desktop\2Paper_result_neuAusfuehrung\plots_xgb\plotsdescriptive_analysis_withoutTitle.png', dpi=500)

#- ----------------------------------------------


# 1. plot true class dist
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)

ax1.hist(ya, alpha = 0.4, density=True) # true values
for i in grida:
    ax1.axvline(i, linestyle='--', color='black', alpha=0.6)
ax1.hist(ya_df.val, bins=grida, alpha=0.4, density=True) # given grid with true values
ax1.plot(xa_gamma, pdf_fitteda, color='lightblue', label='orig') # fitted gamma dist on true values
ax1.set_xlim(0,800); ax1.set_ylim(0,0.007)


ax2.hist(ya_rs, alpha = 0.4, density=True, color='green') # shifted values
for i in grida:
    ax2.axvline(i, linestyle='--', color='black', alpha=0.6) # grid
ax2.hist(ya_rs, bins=grida, alpha=0.4, density=True, color='purple')
ax2.plot(xa_gamma, ya_gamma, color='green', label='orig') # verteilung
ax2.set_xlim(0,800); ax2.set_ylim(0,0.007)

plt.legend()
plt.show()
