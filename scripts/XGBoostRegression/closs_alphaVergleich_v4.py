#
# https://machinelearningmastery.com/results-for-standard-classification-and-regression-machine-learning-datasets/

# mit early stopping and CV and hypertuning
# split train, val und test
# normalized

# %%
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor, XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# import packages for hyperparameters tuning
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, early_stop
from time import time
from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score

# from utils.gradientBoosting import GradientBooster

# %%
# regression mlp model for the abalone dataset
# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/'
data_set = 'ames.csv'
# data_set = 'housing.csv'

#%%
if data_set == 'ames.csv':
    ames = pd.read_csv(r'C:\Users\AdminRBG.SSt\source\repos\amsOS_PPC_FEST\Real_dummyData\AmesHousing.csv')
    y = ames.SalePrice.copy() / 1000.0
    X = ames.drop('SalePrice', axis=1).copy()
    # Remove columns with NaN or Inf values
    X = X.drop(columns=['Lot Frontage', 'Garage Yr Blt', 'Mas Vnr Area'], axis=1)
    X = pd.get_dummies(X)
    X.columns = X.columns.astype(str)

else:
    dataframe = pd.read_csv(fr'{url}/{data_set}', header=None)
    # split into input (X) and output (y) variables
    X, y = dataframe.iloc[:, :-1], dataframe.iloc[:, -1]
    n_features = X.shape[1]

#%%
# define grid
if data_set == 'ames.csv':
    grid = [0,100,250,500,800] # grid for housing
    # add outlier area for learning process
else:
    grid = [0, 10, 25, 45, 60] # grid for housing

if data_set == 'ames.csv':
    grid = [-200,0,100,250,500,800,1000] # grid for housing
    # add outlier area for learning process
else:
    grid = [-20, 0, 10, 25, 45, 60, 100] # grid for housing
class_names = ['cheap','moderate','expensiv','luxurious',' ']
grid_arr = np.array([grid, np.array(range(1,len(grid)+1)).tolist()]) # , class_names])

y_df = pd.DataFrame({'val': y}) # all data
y_df['trCL'] = (grid_arr[1][np.digitize(y_df.val, grid_arr[0])].astype(int)-1).astype(str)

# -------------
# x range gamma
x_gamma = np.linspace(0, max(grid), int(y_df.val.shape[0]*1.1))
# get gamma dist of y
param = stats.gamma.fit(y_df.val, floc=0)
pdf_fitted = stats.gamma.pdf(x_gamma, *param)

if data_set == 'ames.csv':
    a = 6; b = 40; c = 800
else:
    a = 3; b = 9; c = 80
    # a = 3; b = 7; c = 60

# define expertise as gamma dist
y_gamma = stats.gamma.pdf(x_gamma, a, 0, b) 
# random samples from distribution
rng = np.random.default_rng(123) # <---------------------------------------- SEED
y_rs = rng.gamma(a, b, size=int(y_df.val.shape[0]*1.1))

# add gen-data to true data
df = pd.concat([X, y_df], axis=1) # split data together to stratify with trCL and genCL
df = df.sort_values(by='val')

# filter for random samples in given range
geny = y_rs[y_rs > grid[1]]
geny = geny[geny < grid[-2]][:len(y)] # reduce sample size to the same size as y and without values out of given x-Grid-range
geny = np.sort(geny)

# add gen data to dataframe
df['geny'] = geny
df['genCL'] = (grid_arr[1][np.digitize(df.geny, grid_arr[0])].astype(int)-1).astype(str)

# show overlapping ratio
temp = pd.DataFrame()
temp['trCL'] = df.trCL.astype(str)
temp['genCL'] = df.genCL.astype(str)
predict = temp.groupby('genCL').count()
true_val = temp.groupby('trCL').count()
diff = pd.concat([predict, true_val], ignore_index=True, axis=1).fillna(0)
diff['argmin'] =  diff.apply(lambda row: row[np.argmin(row)], axis=1)
diff['argmin_per'] = diff.argmin / diff[0].sum()
np.round(diff.argmin_per.sum(),4) * 100

fig, (ax1, ax2) = plt.subplots(2)
ax1.hist(df.val, density=True, alpha=0.4)
ax1.hist(df.geny, density=True, alpha=0.4)
ax2.hist(df.trCL, density=True, alpha=0.4)
ax2.hist(df.genCL, density=True, alpha=0.4)
# plt.show()

#%% 
# make key for stratified split
df['key'] = df.genCL # df.groupby('key').count() # check for class combinations
df_train, df_test = train_test_split(df, test_size=0.33, random_state=42, stratify=df['key']) # <---------------------------------------- SEED

#%% 
# https://datascience.stackexchange.com/questions/74780/how-to-implement-custom-loss-function-that-has-more-parameters-with-xgbclassifie
# functions

def COR_loss(y_df, alpha): # possibly add subsample ratio 

    y_df = np.array(y_df)

    def customloss(y_val, y_pred):        # l(y_val, y_pred) = (y_val-y_pred)**2

        y_true = y_val # = y_df[:,0] # 
        yclf = y_df[:,1]

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
        grad = 2 * alpha * (y_pred - y_true) + 2 * (1-alpha) * temp.grad2
        hess = 2 * np.ones(len(y_true))

        return grad, hess

    return customloss


def f1_OptLoss(y_val, y_pred):

    try:    
        class_true = (grid_arr[1][np.digitize(y_val, grid_arr[0])].astype(int)-1)
        class_pred = (grid_arr[1][np.digitize(y_pred, grid_arr[0])].astype(int)-1)
        return 1-f1_score(class_true, class_pred, average='weighted')
    except IndexError:
        print('Prediction out of grid range for F1_optloss')
        return 1

def objective(space):

    eval_metric1=f1_OptLoss

    mod=XGBRegressor(
                    objective = loss_fct_val,
                    n_estimators = int(space['n_estimators']), max_depth = int(space['max_depth']), # gamma = space['gamma'],
                    reg_lambda=int(space['reg_lambda']), # reg_alpha = int(space['reg_alpha']),
                    min_child_weight=int(space['min_child_weight']), colsample_bytree=int(space['colsample_bytree']),
                    learning_rate=space['learning_rate'], # subsample=space['subsample'],
                    early_stopping_rounds=early_stop_no, eval_metric=eval_metric1 # 'mae'
                    )
    
    evaluation = [(X_train, y_train.val), (X_val, y_val.val)]
    mod.fit(X_train, y_train.val, eval_set=evaluation, verbose=False)
    pred = mod.predict(X_val)

    return {'loss': f1_OptLoss(y_val.val, pred), 'status': STATUS_OK}

# same for classification
def objectiveCLF(space):

    obj = eval_ = 'mlogloss'

    mod=XGBClassifier(
                    objective = obj, # 'mlogloss', # 'binary:logistic', # 'multi:softmax', # 'roc_auc' #  
                    n_estimators = int(space['n_estimators']), max_depth = int(space['max_depth']), # gamma = space['gamma'],
                    # reg_lambda=int(space['reg_lambda']), # reg_alpha = int(space['reg_alpha']), 
                    min_child_weight=int(space['min_child_weight']),
                    colsample_bytree=int(space['colsample_bytree']),
                    learning_rate=space['learning_rate'], # subsample=space['subsample'],
                    eval_metric=eval_, early_stopping_rounds=5
                    )
    
    sample_weights = compute_sample_weight(
        class_weight='balanced',
        y=y_train_clf['genCL'] #provide your own target name
    )

    evaluation = [(X_train, y_train_clf.genCL), (X_val, y_val_clf.genCL)]
    mod.fit(X_train, y_train_clf.genCL, sample_weight=sample_weights, eval_set=evaluation, verbose=False)
    pred = mod.predict(X_val)
    f1 = f1_score(y_val_clf.genCL, pred, average='weighted')
    return {'loss': 1-f1, 'status': STATUS_OK}


# --------------------------------------------------------------------------------
# separate dataset in train and validation - 5-fold_CV
df_train_shuffled = df_train.sample(frac=1, random_state=123).reset_index(drop=True).copy()
df_test_shuffled = df_test.sample(frac=1, random_state=123).reset_index(drop=True).copy()

y_test = df_test_shuffled[['val','genCL']].copy()
X_test = df_test_shuffled.drop(['val','trCL','geny','genCL','key'], axis=1).copy()


#%%
# Model 

dict_hyperparams = dict()
dict_zwischenRes = dict()

early_stop_no = 20

alpha_list = [1,.75,.5,.3,.1,0.05,0,'clf']
# alpha_list = ['clf']

# Dicts
mae_insample_cv = []
acc_insample_cv = []
f1_insample_cv = []
mae_outsample_cv = []
acc_outsample_cv = []
f1_outsample_cv = []
mae_outsample_cv_std = []
acc_outsample_cv_std = []
f1_outsample_cv_std = []
ct_cv = []


for alpha in alpha_list:
    
    if alpha != 'clf':

        if data_set == 'housing.csv':
            space={'max_depth': hp.quniform("max_depth", 3, 10, 1), # 20
                    'gamma': hp.uniform('gamma', 1, 9), #
                    # 'reg_alpha' : hp.quniform('reg_alpha', 0, 80, 1),
                    'reg_lambda' : hp.uniform('reg_lambda', 0, 1),
                    'colsample_bytree' : hp.uniform('colsample_bytree', 0.5, 1),
                    'min_child_weight' : hp.quniform('min_child_weight', 0, 15, 1), #
                    'n_estimators': hp.quniform('n_estimators', 50, 1200, 50), # int(150/(alpha+0.1)), 20), # 1000 #  int(150/(alpha+0.1))
                    'learning_rate': hp.uniform('learning_rate', 0.01, 0.2), #
                    # 'subsample': hp.uniform('subsample', 0.5, 0.8), # 1
                    'seed': 0
                }
        elif data_set == 'ames.csv':
            space={'max_depth': hp.quniform("max_depth", 3, 10, 1), # 20
                    'gamma': hp.uniform('gamma', 1, 9), #
                    # 'reg_alpha' : hp.quniform('reg_alpha', 0, 80, 1),
                    'reg_lambda' : hp.uniform('reg_lambda', 0, 1),
                    'colsample_bytree' : hp.uniform('colsample_bytree', 0.5, 1),
                    'min_child_weight' : hp.quniform('min_child_weight', 0, 15, 1), #
                    'n_estimators': hp.quniform('n_estimators', 50, 1200, 50), # int(150/(alpha+0.1)), 20), # 1000 #  int(150/(alpha+0.1))
                    'learning_rate': hp.uniform('learning_rate', 0.01, 0.2), #
                    # 'subsample': hp.uniform('subsample', 0.5, 0.8), # 1
                    'seed': 0
                }

        mae_insample = []; mae_outsample = []
        acc_insample = []; acc_outsample = []
        f1_insample = []; f1_outsample = []
        ct = []

        for k in range(5): # range(3): # 
            
            df_train, df_val = train_test_split(df_train_shuffled, test_size=0.2, stratify=df_train_shuffled['key'])
            y_train_and_val = df_train_shuffled[['val','genCL']].copy().reset_index(drop=True)
            X_train_and_val = df_train_shuffled.drop(columns=['val','trCL','geny','genCL','key']).copy().reset_index(drop=True)
            y_train = df_train[['val','genCL']].copy().reset_index(drop=True)
            X_train = df_train.drop(columns=['val','trCL','geny','genCL','key']).copy().reset_index(drop=True)
            y_val = df_val[['val','genCL']].copy().reset_index(drop=True)
            X_val = df_val.drop(columns=['val','trCL','geny','genCL','key']).copy().reset_index(drop=True)

            if alpha == 1:
                loss_fct_val = 'reg:squarederror'
                loss_fct = 'reg:squarederror'
            else:
                loss_fct_val = COR_loss(y_df=y_train, alpha=alpha)
                loss_fct = COR_loss(y_df=y_train_and_val, alpha=alpha)

            start = time()
            # XGBoost
            trials = Trials() # use validation data 
            best_hyperparams = fmin(fn = objective, space = space, algo = tpe.suggest, max_evals = 150, trials = trials, early_stop_fn = early_stop.no_progress_loss(early_stop_no))
            best_hyperparams['max_depth'] = int(best_hyperparams['max_depth']); best_hyperparams['n_estimators'] = int(best_hyperparams['n_estimators'])

            dict_hyperparams[f'alpha={alpha}_k={k}'] = best_hyperparams
            
            mod = XGBRegressor(objective=loss_fct,**best_hyperparams)
            # mod = XGBRegressor(objective=loss_fct)
            mod.fit(X_train_and_val, y_train_and_val.val, verbose=False)
            
            pred_ins = mod.predict(X_train_and_val); pred_ins_df = pd.DataFrame({'Insample': pred_ins})
            pred_outs = mod.predict(X_test); pred_outs_df = pd.DataFrame({'Outsample': pred_outs})

            # pointestimation --sort--> pred-classes
            classes = np.digitize(pred_ins_df.Insample, grid_arr[0]); pred_ins_df['classes'] = (grid_arr[1][classes].astype(int)-1).astype(str).copy()
            classes = np.digitize(pred_outs_df.Outsample, grid_arr[0]); pred_outs_df['classes'] = (grid_arr[1][classes].astype(int)-1).astype(str).copy()
            ct.append(time()-start)
            # ----------------------------------------------------
            # Metrics insample
            acc_insample.append(accuracy_score(y_train_and_val.genCL.astype(str), pred_ins_df.classes)*100) # Accuracy
            mae_insample.append(mean_absolute_error(pred_ins, y_train_and_val.val)) # Mae
            f1_insample.append(f1_score(y_train_and_val.genCL.astype(str), pred_ins_df.classes, average='weighted')*100) # f1 score
            # ----------------------------------------------------
            # Metrics outsample
            acc_outsample.append(accuracy_score(y_test.genCL.astype(str), pred_outs_df.classes)*100) # Accuracy
            mae_outsample.append(mean_absolute_error(pred_outs, y_test.val)) # Mae
            f1_outsample.append(f1_score(y_test.genCL.astype(str), pred_outs_df.classes, average='weighted')*100) # f1 score
            # ----------------------------------------------------
            dict_zwischenRes[f'alpha={alpha}_k={k}'] = {'f1_in':f1_score(y_train_and_val.genCL.astype(str), pred_ins_df.classes, average='weighted')*100, 'f1_out':f1_score(y_test.genCL.astype(str), pred_outs_df.classes, average='weighted')*100}
            dict_zwischenRes

        # mean of cv
        mae_insample_cv.append(np.mean(mae_insample)); mae_outsample_cv.append(np.mean(mae_outsample)); mae_outsample_cv_std.append(np.std(mae_outsample))
        acc_insample_cv.append(np.mean(acc_insample)); acc_outsample_cv.append(np.mean(acc_outsample)); acc_outsample_cv_std.append(np.std(acc_outsample))
        f1_insample_cv.append(np.mean(f1_insample)); f1_outsample_cv.append(np.mean(f1_outsample)); f1_outsample_cv_std.append(np.std(f1_outsample))
        ct_cv.append(np.mean(ct)) #; print(f'Finish: alpha = {alpha}')

    else: # =='clf'

        # Tuning domain
        if data_set == 'housing.csv':
            
            space_clf={'max_depth': hp.quniform("max_depth", 3, 10, 1), # 20
                # 'gamma': hp.uniform('gamma', 1, 9), #
                # 'reg_alpha' : hp.quniform('reg_alpha', 0,80,1),
                # 'reg_lambda' : hp.uniform('reg_lambda', 0, 1),
                'colsample_bytree' : hp.uniform('colsample_bytree', 0.5, 1),
                'min_child_weight' : hp.quniform('min_child_weight', 0, 15, 1), #
                'n_estimators': hp.quniform('n_estimators', 50, 1000, 50), # 1000
                'learning_rate': hp.uniform('learning_rate', 0.01, 0.2), #
                # 'subsample': hp.uniform('subsample', 0.5, 0.8), # 1
                'seed': 0
            }

        elif data_set == 'ames.csv':
            
            space_clf={'max_depth': hp.quniform("max_depth", 3, 10, 1), # 20
                # 'gamma': hp.uniform('gamma', 1, 9), #
                # 'reg_alpha' : hp.quniform('reg_alpha', 0,80,1),
                # 'reg_lambda' : hp.uniform('reg_lambda', 0, 1),
                'colsample_bytree' : hp.uniform('colsample_bytree', 0.5, 1),
                'min_child_weight' : hp.quniform('min_child_weight', 0, 15, 1), #
                'n_estimators': hp.quniform('n_estimators', 50, 1000, 50), # 1000
                'learning_rate': hp.uniform('learning_rate', 0.01, 0.2), #
                # 'subsample': hp.uniform('subsample', 0.5, 0.8), # 1
                'seed': 0
            }
                        
        mae_insample = []; mae_outsample = []
        acc_insample = []; acc_outsample = []
        f1_insample = []; f1_outsample = []
        ct = []

        for k in range(5): # range(3): # 

            df_train, df_val = train_test_split(df_train_shuffled, test_size=0.2, stratify=df_train_shuffled['key'])
            y_train_and_val = df_train_shuffled[['val','genCL']].copy().reset_index(drop=True)
            X_train_and_val = df_train_shuffled.drop(columns=['val','trCL','geny','genCL','key']).copy().reset_index(drop=True)
            y_train = df_train[['val','genCL']].copy().reset_index(drop=True)
            X_train = df_train.drop(columns=['val','trCL','geny','genCL','key']).copy().reset_index(drop=True)
            y_val = df_val[['val','genCL']].copy().reset_index(drop=True)
            X_val = df_val.drop(columns=['val','trCL','geny','genCL','key']).copy().reset_index(drop=True)

            # Model ---------------
            start = time()

            y_train_clf = y_train.copy()
            y_test_clf = y_test.copy()
            y_val_clf = y_val.copy()
            y_train_and_val_clf = y_train_and_val.copy()

            le = LabelEncoder().fit(y_train_and_val_clf.genCL)
            y_train_clf.genCL = le.transform(y_train_clf.genCL)
            y_val_clf.genCL = le.transform(y_val_clf.genCL)
            y_test_clf.genCL = le.transform(y_test_clf.genCL)
            y_train_and_val_clf.genCL = le.transform(y_train_and_val_clf.genCL)

            trials = Trials()
            best_hyperparams = fmin(fn = objectiveCLF, space = space_clf, algo = tpe.suggest, max_evals = 100, trials = trials, early_stop_fn = early_stop.no_progress_loss(5))
            best_hyperparams['max_depth'] = int(best_hyperparams['max_depth']); 
            best_hyperparams['n_estimators'] = int(best_hyperparams['n_estimators'])

            dict_hyperparams[f'alpha={alpha}_k={k}'] = best_hyperparams

            mod = XGBClassifier(**best_hyperparams)
            # mod = XGBClassifier()
            sample_weights = compute_sample_weight(
                class_weight='balanced',
                y= y_train_and_val_clf['genCL'] #provide your own target name
            )
            mod.fit(X_train_and_val, y_train_and_val_clf.genCL, sample_weight=sample_weights, verbose=False) # change here to train_and_val           
            y_train_and_val_clf['predCL'] = mod.predict(X_train_and_val)
            y_test_clf['predCL'] = mod.predict(X_test)

            ct.append(time()-start)
            # ----------------------------------------------------
            # Metrics insample
            acc_insample.append(accuracy_score(y_train_and_val_clf.genCL, y_train_and_val_clf.predCL)*100) # Accuracy
            mae_insample.append(np.NaN) # MaE insample
            f1_insample.append(f1_score(y_train_and_val_clf.genCL, y_train_and_val_clf.predCL, average='weighted')*100) # f1 score
            # ----------------------------------------------------
            # Metrics outsample
            mae_outsample.append(np.NaN) # MaE outsample
            f1_outsample.append(f1_score(y_test_clf.genCL, y_test_clf.predCL, average='weighted')*100) # f1 score
            acc_outsample.append(accuracy_score(y_test_clf.genCL, y_test_clf.predCL)*100) # Accuracy
            # ----------------------------------------------------

        # mean of cv
        mae_insample_cv.append(np.mean(mae_insample)); mae_outsample_cv.append(np.mean(mae_outsample)); mae_outsample_cv_std.append(np.std(mae_outsample))
        acc_insample_cv.append(np.mean(acc_insample)); acc_outsample_cv.append(np.mean(acc_outsample)); acc_outsample_cv_std.append(np.std(acc_outsample))
        f1_insample_cv.append(np.mean(f1_insample)); f1_outsample_cv.append(np.mean(f1_outsample)); f1_outsample_cv_std.append(np.std(f1_outsample))
        ct_cv.append(np.mean(ct))
        print(f'Finish: alpha = clf')

#

temp = pd.DataFrame({
                    'mae_train': np.round(mae_insample_cv,3), 'mae_test': np.round(mae_outsample_cv,3), 'mae_test_sd': np.round(mae_outsample_cv_std,3),
                    'acc_train': np.round(acc_insample_cv,2), 'acc_test': np.round(acc_outsample_cv,2), 'acc_test_sd': np.round(acc_outsample_cv_std,2), 
                    'f1_train': np.round(f1_insample_cv,2), 'f1_test': np.round(f1_outsample_cv,2), 'f1_test_sd': np.round(f1_outsample_cv_std,2), 
                    'ct': np.round(ct_cv,1)
                    }) #, index=alpha_list)

print(temp.T)

# %%
temp.T.to_excel(fr'C:\Users\AdminRBG.SSt\Desktop\2Paper_result_neuAusfuehrung\res_{data_set}.xlsx')

a = pd.DataFrame(dict_hyperparams)
a.to_excel(fr'C:\Users\AdminRBG.SSt\Desktop\2Paper_result_neuAusfuehrung\params_{data_set}.xlsx')
