#%%
import numpy as np
import pandas as pd
import matplotlib
import scipy.stats as stats
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score

font = {'size': 8}
matplotlib.rc('font', **font)
plt.rcParams.update({'font.family':'Times New Roman', 'text.color' : "black", 'axes.labelcolor' : "black"})


#%% model class
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

            # mae loss
            dw = (1/n_samples) * np.dot(X.T, (y_pred-y))
            db = (1/n_samples) * np.sum(y_pred-y)

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db
    
    def fit_COR(self, X, y):

        y_true = y[:,0]
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
sample_size = 1000
X, y = datasets.make_regression(n_samples=sample_size, n_features=1, noise=20, random_state=4)

#% find gamma dist
grid_def = [-200,-50,50,150,200]
grid = [-300,-200,-50,50,150,200,300]
grid_arr = np.array([grid, np.array(range(1,len(grid)+1)).tolist()]) # , class_names])

y_df = pd.DataFrame({'val': y}) # all data
y_df['trCL'] = (grid_arr[1][np.digitize(y_df.val, grid_arr[0])].astype(int)-1).astype(str)

# -------------
# gamma of real data
x_norm = np.linspace(min(grid), max(grid), int(sample_size*1.1))
param = stats.norm.fit(y_df.val, floc=0); # pdf_fitted = stats.norm.pdf(x_norm, *param)

# new dist gamma
mu = 5; sigma=65
y_norm = stats.norm.pdf(x_norm, mu, sigma)
# random samples from distribution
seed_no = 341 # 566
rng = np.random.default_rng(seed_no)
y_rs = rng.normal(mu, sigma, size=int(sample_size*1.1))

# add gen-data to true data
df = pd.concat([pd.DataFrame(X), y_df], axis=1) # split data together to stratify with trCL and genCL
df = df.sort_values(by='val').reset_index(drop=True)

geny = y_rs[y_rs > grid_def[0]]
geny = geny[geny < grid_def[-1]][:len(y)] # reduce sample size to the same size as y and without values out of given x-Grid-range
geny = np.sort(geny)

# add gen data to dataframe
df['geny'] = geny
df['genCL'] = (grid_arr[1][np.digitize(df.geny, grid_arr[0])].astype(int)-1).astype(str)

# make key for stratified split
df['key'] = df.genCL 
df.groupby('key').count() # check for class combinations


# plt.plot(df.val, df.geny, '-')
# plt.show()

# # show data
# fig, (ax1, ax2) = plt.subplots(2)
# ax1.hist(geny, bins=grid, density=True)
# ax2.hist(y_df.val, alpha=0.4, density=True)
# ax2.hist(y_df.val, bins=grid, alpha=0.4, density=True)
# plt.show()


#%%
plot_true = False

alpha = 0.1 # alpha_list = [1,.9,.8,.7,.6,.5,.4,.3,.2,.1,0]
lr = 0.1 # lr_list = [.01,.1,.2,.3,.4,.5,.6,.7,.8,.9,1]
iter_list =  [1,10,50,100,200,300,500,1000,1500,1800,2000,4000]
# iter_list =  [1,100,300,500] #,4000]

# dicts
dict_mae_def = []; dict_f1_def = []; dict_acc_def = []
dict_mae = []; dict_f1 = []; dict_acc = []

m_dict_mae_def = []; m_dict_f1_def = []; m_dict_acc_def = []
m_dict_mae = []; m_dict_f1 = []; m_dict_acc = []

std_dict_mae_def = []; std_dict_f1_def = []
std_dict_mae = []; std_dict_f1 = []

i = 100
for i in iter_list:

    for k in range(3):

        # seed_list = [94,45,80,164,32]
        seed_list = [1465,45984,487,4897,88]
        df_train, df_test = train_test_split(df, test_size=0.33, random_state=seed_list[k], stratify=df['key']) # random_state=42, 

        X_train = df_train[0]; X_train = np.array(X_train).reshape(len(X_train),1)
        y_train = df_train[['val','genCL']]; y_train = np.array(y_train)

        X_test = df_test[0]; X_test = np.array(X_test).reshape(len(X_test),1)
        y_test = df_test[['val','genCL']]; y_test = np.array(y_test)

        # Model
        mod = LinearRegression(n_iters=i, lr=lr, alpha=alpha)

        # Baseline model training
        mod.fit(X_train, y_train[:,0])
        pred_train_default = mod.predict(X_train)
        pred_default = mod.predict(X_test)

        # second model training
        mod.fit_COR(X_train, y_train)
        pred_train = mod.predict(X_train)
        pred = mod.predict(X_test)

        # find class of pred
        pred_train_defaultCL = (grid_arr[1][np.digitize(pred_train_default, grid_arr[0])].astype(int)-1)
        predCL_train = (grid_arr[1][np.digitize(pred_train, grid_arr[0])].astype(int)-1)
        pred_defaultCL = (grid_arr[1][np.digitize(pred_default, grid_arr[0])].astype(int)-1)
        predCL = (grid_arr[1][np.digitize(pred, grid_arr[0])].astype(int)-1)
        y_trainCL = y_train[:,1].astype(int)
        y_testCL = y_test[:,1].astype(int) # change y_test class to int

        # Evalmetrics --------------------------------------------------------
        # mae
        dict_mae_def.append(mean_absolute_error(pred_default, y_test[:,0]))
        dict_mae.append(mean_absolute_error(pred, y_test[:,0]))
        # f1-score
        dict_f1_def.append(f1_score(y_testCL.astype(str), pred_defaultCL.astype(str), average='weighted')*100)
        dict_f1.append(f1_score(y_testCL.astype(str), predCL.astype(str), average='weighted')*100)
        # accuracy
        dict_acc_def.append(accuracy_score(y_testCL.astype(str), pred_defaultCL.astype(str))*100)
        dict_acc.append(accuracy_score(y_testCL.astype(str), predCL.astype(str))*100)

        a_train = f1_score(y_trainCL.astype(str), pred_train_defaultCL.astype(str), average='weighted')*100
        b_train = f1_score(y_trainCL.astype(str), predCL_train.astype(str), average='weighted')*100
        a = f1_score(y_testCL.astype(str), pred_defaultCL.astype(str), average='weighted')*100
        b = f1_score(y_testCL.astype(str), predCL.astype(str), average='weighted')*100

        print(f'Iter: {i}, Run: {k}')
        print(f'f1_def_train: {np.round(a_train,2)}, f1: {np.round(b_train,2)}')
        print(f'f1_def:       {np.round(a,2)}, f1: {np.round(b,2)}')

    m_dict_mae_def.append(np.mean(dict_mae_def)); m_dict_f1_def.append(np.mean(dict_f1_def)); m_dict_acc_def.append(np.mean(dict_acc_def))
    m_dict_mae.append(np.mean(dict_mae)); m_dict_f1.append(np.mean(dict_f1)); m_dict_acc.append(np.mean(dict_acc))
    std_dict_mae_def.append(np.std(dict_mae_def)); std_dict_f1_def.append(np.std(dict_f1_def))
    std_dict_mae.append(np.std(dict_mae)); std_dict_f1.append(np.std(dict_f1))



#%%
res = pd.DataFrame({'Iteration': iter_list,
                    'MAE (BL)': m_dict_mae_def,
                    'MAE (BL) std': std_dict_mae_def,
                    'MAE': m_dict_mae,
                    'MAE std': std_dict_mae,
                    'F1_score (BL)': m_dict_f1_def,
                    'F1_score (BL) std': std_dict_f1_def,
                    'F1_score': m_dict_f1,
                    'F1_score std': std_dict_f1,
                    'Acc (BL)': m_dict_acc_def,
                    'Acc': m_dict_acc})

#%%
res.to_excel(fr'C:\Users\AdminRBG.SSt\Desktop\2Paper_result_neuAusfuehrung\res_itervar_alpha={alpha}_lr={lr}.xlsx')


alpha = 0.1; lr = 0.8
res = pd.read_excel(fr'C:\Users\AdminRBG.SSt\Desktop\2Paper_result_neuAusfuehrung\linReg\res_itervar_alpha={alpha}_lr={lr}.xlsx')
# %%
dropAfter = -3
z = 1.96

fig, (ax1,ax2) = plt.subplots(2)
ax1.plot(res['Iteration'][:dropAfter], res['MAE (BL)'][:dropAfter], 'x', linestyle='-')
ax1.fill_between(
    res['Iteration'][:dropAfter].ravel(),
    res['MAE (BL)'][:dropAfter] - z* res['MAE (BL) std'][:dropAfter],
    res['MAE (BL)'][:dropAfter] + z* res['MAE (BL) std'][:dropAfter],
    color="tab:blue",
    alpha=0.2,
)
ax1.set_ylim(8.5,31)
ax1.plot(res['Iteration'][:dropAfter], res['MAE'][:dropAfter], 'x', linestyle='-')
ax1.set_ylabel('MAE')
ax1.fill_between(
    res['Iteration'][:dropAfter].ravel(),
    res['MAE'][:dropAfter] - z* res['MAE std'][:dropAfter],
    res['MAE'][:dropAfter] + z* res['MAE std'][:dropAfter],
    color="tab:orange",
    alpha=0.2,
)

ax2.plot(res['Iteration'][:dropAfter], res['F1_score (BL)'][:dropAfter], 'x', linestyle='-')
ax2.fill_between(
    res['Iteration'][:dropAfter].ravel(),
    res['F1_score (BL)'][:dropAfter] - z* res['F1_score (BL) std'][:dropAfter],
    res['F1_score (BL)'][:dropAfter] + z* res['F1_score (BL) std'][:dropAfter],
    color="tab:blue",
    alpha=0.2,
)
ax2.set_ylim(22,88.5)
ax2.plot(res['Iteration'][:dropAfter], res['F1_score'][:dropAfter], 'x', linestyle='-')
ax2.fill_between(
    res['Iteration'][:dropAfter].ravel(),
    res['F1_score'][:dropAfter] - z* res['F1_score std'][:dropAfter],
    res['F1_score'][:dropAfter] + z* res['F1_score std'][:dropAfter],
    color="tab:orange",
    alpha=0.2,
)
ax2.set_xlabel('# iteration'); ax2.set_ylabel('F1-score')

plt.suptitle(fr'fixed parameter: $\alpha={alpha}$, $lr={lr}$')
plt.tight_layout()

# plt.show()


plt.savefig(fr'C:\Users\AdminRBG.SSt\Desktop\2Paper_result_neuAusfuehrung\linReg\plots\GenEx_itervar_alpha={alpha}_lr={lr}.png', dpi=500)

# %%
