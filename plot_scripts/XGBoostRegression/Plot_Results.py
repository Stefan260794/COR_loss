#%%
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

font = {'size': 8}
matplotlib.rc('font', **font)
plt.rcParams.update({'font.family':'Times New Roman', 'text.color' : "black", 'axes.labelcolor' : "black"})



df = pd.read_excel(r'C:\Users\AdminRBG.SSt\Desktop\paper2_auswertung1_2Versuch.xlsx')
df = df.iloc[:-1,:]
df_2 = pd.read_excel(r'C:\Users\AdminRBG.SSt\Desktop\paper2_auswertung1_2Versuch_ames.xlsx')
df_2 = df_2.iloc[:-1,:]

clf = pd.read_excel(r'C:\Users\AdminRBG.SSt\Desktop\paper2_auswertung1_2Versuch.xlsx')
clf = clf.iloc[-1,:]
clf_2 = pd.read_excel(r'C:\Users\AdminRBG.SSt\Desktop\paper2_auswertung1_2Versuch_ames.xlsx')
clf_2 = clf_2.iloc[-1,:]



fig, ((ax1, ax2), (ax3,ax4)) = plt.subplots(2,2)

#mae
ax1.plot(df.index, df.mae, linestyle='-', marker='o', markersize=3.5, color='darkorange') # mae bh
ax1.fill_between(
    df.index.ravel(),
    df.mae - 1.96*df.mae_std,
    df.mae + 1.96*df.mae_std,
    color="tab:orange",
    alpha=0.2,
)
ax1.axvline(4, linestyle='--', color='grey')
ax1.set_xticks(np.arange(0,7, 1))
ax1.set_xticklabels(df.alpha)
ax1.set_title('Boston Housing')
ax1.set_ylabel('MAE')
ax1.set_xlim(0,6)

ax2.plot(df_2.index, df_2.mae, linestyle='-', marker='o', markersize=3.5, color='darkorange') # mae bh
ax2.fill_between(
    df_2.index.ravel(),
    df_2.mae - 1.96*df_2.mae_std,
    df_2.mae + 1.96*df_2.mae_std,
    color="tab:orange",
    alpha=0.2,
)
ax2.axvline(4, linestyle='--', color='grey')
ax2.set_xticks(np.arange(0,7, 1))
ax2.set_xticklabels(df_2.alpha)
ax2.set_title('Ames Housing')
ax2.set_xlim(0,6)

# f1score
ax3.plot(df.index, df.f1, linestyle='-', marker='o', markersize=3.5) # mae bh
ax3.fill_between(
    df.index.ravel(),
    df.f1 - 1.96*df.f1_std,
    df.f1 + 1.96*df.f1_std,
    color="tab:blue",
    alpha=0.2,
)
ax3.axhline(clf.f1, xmin=0, xmax=6, linestyle='-.', color='green')
ax3.axvline(4, linestyle='--', color='grey')
ax3.fill_between(
    clf.index.ravel(),
    clf.f1 - 1.96*clf.f1_std,
    clf.f1 + 1.96*clf.f1_std,
    color="tab:green",
    alpha=0.2,
)
ax3.set_xticks(np.arange(0,7, 1))
ax3.set_xticklabels(df.alpha)
ax3.set_xlabel(r'$\alpha$ values')
ax3.set_ylabel('F1-score')
ax3.set_xlim(0,6)

ax4.plot(df_2.index, df_2.f1, linestyle='-', marker='o', markersize=3.5) # mae bh
ax4.fill_between(
    df_2.index.ravel(),
    df_2.f1 - 1.96*df_2.f1_std,
    df_2.f1 + 1.96*df_2.f1_std,
    color="tab:blue",
    alpha=0.2,
)
ax4.axhline(clf_2.f1, xmin=0, xmax=6, linestyle='-.', color='green')
ax4.axvline(4, linestyle='--', color='grey')
ax4.fill_between(
    clf_2.index.ravel(),
    clf_2.f1 - clf_2.f1_std,
    clf_2.f1 + clf_2.f1_std,
    color="tab:green",
    alpha=0.2,
)
ax4.axvline(4, linestyle='--', color='grey')
ax4.set_xticks(np.arange(0,7, 1))
ax4.set_xticklabels(df_2.alpha)
ax4.set_xlabel(r'$\alpha$ values')
ax4.set_xlim(0,6)

plt.tight_layout()
# plt.show()

plt.savefig(fr'C:\Users\AdminRBG.SSt\Desktop\2Paper_result_neuAusfuehrung\plots_xgb\Auswertung_1_2version.png', dpi=500)


