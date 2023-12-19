# %% 
import warnings
warnings.filterwarnings("ignore")
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import pingouin as pg
import scipy as sp
import matplotlib.pyplot as plt
from nilearn.input_data import NiftiLabelsMasker
from nilearn.plotting import plot_surf
from nctpy.utils import matrix_normalization
from nctpy.metrics import ave_control
from scipy.stats import pearsonr, ttest_rel, ttest_1samp
from neuromaps.datasets import fetch_annotation, fetch_fslr
from neuromaps.images import relabel_gifti, dlabel_to_gifti
from neuromaps.stats import compare_images
from neuromaps.parcellate import Parcellater
from neuromaps.transforms import fslr_to_fslr
from neuromaps.nulls import burt2020
from matplotlib.gridspec import GridSpec
from src.utils import get_gordon_palette, get_significance_string
from src.cmaps import sequential_green, custom_coolwarm

# %% General variables

dataset = 'hcp'
atlas = 'Schaefer400'
root_dir = './data/'
dataset_dir = root_dir + f'{dataset}/'
output_dir = f'./results/{dataset}/'

if dataset == 'tum':
    subjects = sorted([fn.split('/')[-1][:7] for fn in glob.glob(dataset_dir + f'sub*timeseries_{atlas}.npy')])
elif dataset == 'hcp':
    subjects = sorted([fn.split('/')[-1][:10] for fn in glob.glob(dataset_dir + f'sub*timeseries_{atlas}.npy')])

if atlas == 'Gordon':
    den = '3mm'
    atlas_vol = root_dir + f'Gordon2016_space-MNI152_den-{den}.nii'
    labels = ['AUD', 'CoP', 'CoPar' ,'DMN', 'DAN', 'FrP', 'None', 'RT', 'SAL','SMH','SMM','VAN','VIS','NOTA']
    atlas_order = pd.read_pickle(root_dir + 'Gordon2016_333_LUT.pkl')
    atlas_fslr = dlabel_to_gifti(root_dir + 'Gordon2016_333_space-fsLR_den-32k.dlabel.nii')
    lab2mod={'AUD':'Unimodal','SMH':'Unimodal','SMM':'Unimodal','VIS':'Unimodal',
             'CoP':'Heteromodal','CoPar':'Heteromodal','DMN':'Heteromodal','FrP':'Heteromodal',
             'DAN':'Heteromodal','RT':'Heteromodal','SAL':'Heteromodal','VAN':'Heteromodal',
             'None':'None','NOTA':'Subthreshold'}

elif atlas == 'Schaefer400':
    den = '2mm'
    atlas_vol = root_dir + f'Schaefer2018_400_7N_space-MNI152_den-{den}.nii.gz'
    labels = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default', 'NOTA']
    atlas_order = pd.read_pickle(root_dir + 'Schaefer2018_400_LUT.pkl')
    atlas_fslr = dlabel_to_gifti(root_dir + 'Schaefer2018_400_7N_space-fsLR_den-32k.dlabel.nii')
    lab2mod = {'Vis':'Unimodal', 'SomMot':'Unimodal', 'DorsAttn':'Heteromodal', 'SalVentAttn':'Heteromodal', 
               'Limbic':'Heteromodal', 'Cont':'Heteromodal', 'Default':'Heteromodal', 'NOTA':'Subthreshold'}

elif atlas == 'Schaefer200':
    den = '2mm'
    atlas_vol = root_dir + f'Schaefer2018_200_7N_space-MNI152_den-{den}.nii.gz'
    labels = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default', 'NOTA']
    atlas_order = pd.read_pickle(root_dir + 'Schaefer2018_200_LUT.pkl')
    atlas_fslr = dlabel_to_gifti(root_dir + 'Schaefer2018_200_7N_space-fsLR_den-32k.dlabel.nii')
    lab2mod = {'Vis':'Unimodal', 'SomMot':'Unimodal', 'DorsAttn':'Heteromodal', 'SalVentAttn':'Heteromodal', 
               'Limbic':'Heteromodal', 'Cont':'Heteromodal', 'Default':'Heteromodal', 'NOTA':'Subthreshold'}

nrois = len(atlas_order)
nlabels = len(labels)
id2net = dict(zip(np.arange(nlabels)+1,labels))
mni = f'./data/MNI152_T1_{den}_brain.nii.gz'

# maskers to parcellate maps
mask = NiftiLabelsMasker(atlas_vol).fit(mni)
surf_masker = Parcellater(atlas_fslr,'fslr')

# %% Prepare average control energy for spatial map comparison
control_df = pd.read_csv(output_dir + f'average-control-energy_subject-level_{atlas}.csv')
avg_ctrl = control_df.groupby('ROI').mean()['E_control'].values

# %% Load modality-wise data
mod_df = pd.read_csv(output_dir + f'average-control-energy_modality_subject-level_{atlas}.csv').drop('E_sub', axis=1)

# rename transition names for readability
trans_dict = {'E_hh':'H→H','E_hu':'H→U','E_uh':'U→H','E_uu':'U→U'}
mod_df = mod_df.rename(columns=trans_dict)

# order by median energy per transition type
trans_list = ['U→U','H→U','U→H','H→H']
col_idx = np.mean(mod_df[trans_list].values,axis=0).argsort()
order = [trans_list[i] for i in col_idx]

# average by subject for each modality
mod_df = mod_df.groupby('subject').mean().reset_index()

# transform to long format for plotting
mod_df = pd.melt(mod_df, value_vars=['U→U','H→U','U→H','H→H'], id_vars='subject', var_name='transition_type')
mod_df = mod_df.sort_values(['subject','transition_type'],ignore_index=True)
hierarchy_dict = {'H→H':'Within', 'U→U':'Within', 'H→U':'Between', 'U→H':'Between'}
mod_df['hierarchy'] = mod_df['transition_type'].map(hierarchy_dict)

x = mod_df[mod_df['hierarchy'] == 'Between']['value']
y = mod_df[mod_df['hierarchy'] == 'Within']['value']
results = pg.mwu(x,y)
p_str = get_significance_string(results['p-val'][0], type='text').upper()
print(f"U = {results['U-val'][0]}")
print(f"P = {p_str}")

# %%
fig = plt.figure(figsize=(6, 5), dpi=200)
gs = GridSpec(1, 2, width_ratios=[1.5, 1])

pal_list = pal_list = sequential_green(return_palette=True)[0:2] + sequential_green(return_palette=True)[3:5]
lower_ylim = 0
alpha = 0.1
lwidth = 1
height_percentage = 0.85
plt.rcParams.update({'font.size': 12})

ax1 = plt.subplot(gs[0])
# sns.barplot(data=mod_df, x='transition_type', y='value', width=0.5, ci='sd', 
#             errwidth=bar_lwidth, order=order, palette=pal_list, ax=ax1)
sns.stripplot(data=mod_df, x='transition_type', y='value', alpha=alpha, 
            linewidth=0.8, order=order, palette=pal_list, ax=ax1)

sns.boxplot(data=mod_df, x='transition_type', y='value', width=0.5, 
            linewidth=0, medianprops={"linewidth":lwidth}, whiskerprops={"linewidth":lwidth},
            order=order, palette=pal_list, ax=ax1)

ax2 = plt.subplot(gs[1])
# sns.barplot(data=mod_df, x='hierarchy', y='value', width=0.8, ci='sd', 
#             errwidth=bar_lwidth, palette='Greys', ax=ax2)
sns.boxplot(data=mod_df, x='hierarchy', y='value', width=0.8, palette='Greys', ax=ax2,
                linewidth=0, medianprops={"linewidth":lwidth}, whiskerprops={"linewidth":lwidth})
# sns.stripplot(data=mod_df, x='hierarchy', y='value', alpha=alpha, 
#                 linewidth=0.8, palette='Greys', ax=ax2)

ax1.set(xlabel=None)
ax1.set_ylabel('Whole-brain TCE [a.u.]')
ax2.spines['left'].set_visible(False)
ax2.tick_params(left=False)
ax2.set(yticklabels=[], xlabel=None, ylabel=None)
sns.despine(ax=ax2, left=True)
fig.supxlabel('Transition type', y=0.05)
plt.subplots_adjust(wspace=0.1)

# Calculate the middle of the two barplots in ax2
middle_x1 = ax2.get_xlim()[0] + (ax2.get_xlim()[1] - ax2.get_xlim()[0]) / 4
middle_x2 = ax2.get_xlim()[0] + 3 * (ax2.get_xlim()[1] - ax2.get_xlim()[0]) / 4

# Draw a horizontal line from the middle of the first bar to the middle of the second bar
ax2.plot([middle_x1, middle_x2], 
            [ax2.get_ylim()[1]*height_percentage, ax2.get_ylim()[1]*height_percentage], 
            color='black', linewidth=1.5)

# Add a star above the line
ax2.annotate(get_significance_string(p), xycoords='data', ha='center', 
                xy=((middle_x1 + middle_x2)/2, ax2.get_ylim()[1]*height_percentage),
                xytext=((middle_x1 + middle_x2)/2, ax2.get_ylim()[1]*height_percentage))

ax1.set_ylim([lower_ylim, ax1.get_ylim()[1]])
ax2.set_ylim([lower_ylim, ax2.get_ylim()[1]])
sns.despine(ax=ax1, trim=True)
plt.tight_layout()
# plt.savefig('./figs/Fig4_ace_modalitites.pdf')

#%% Number of transitions
ntransitions = np.empty((0))
all_mtx = np.empty((2, 2, 0))

for subj_id in subjects:
    targets = np.load(dataset_dir + f'{subj_id}_task-rest_bold_desc-time_labels_{atlas}.npy')
    targets = [*map(id2net.get,targets)]
    mod = [*map(lab2mod.get,targets)]
    # Transition matrix
    mtx = np.zeros((2, 2))
    for ii,current in enumerate(mod):
        if ii==0:
            continue
        previous = mod[ii-1]
        if previous == 'Heteromodal' and current == 'Heteromodal':
            mtx[0,0] += 1
        elif previous == 'Heteromodal' and current == 'Unimodal':
            mtx[0,1] += 1
        elif previous == 'Unimodal' and current == 'Heteromodal':
            mtx[1,0] += 1
        elif previous == 'Unimodal' and current == 'Unimodal':
            mtx[1,1] += 1
        else:
            continue
    ntransitions = np.append(ntransitions,mtx.flatten())
    all_mtx = np.concatenate((all_mtx, mtx[:,:,None]), axis=-1)
mod_df['NTr'] = np.log10(ntransitions)

# %%
ticks = ['Unimodal', 'Heteromodal']
avg_transition = all_mtx.mean(axis=-1).astype(int)
avg_transition = np.flip(avg_transition)
plt.figure(dpi=250)
plt.rcParams.update({'font.size': 14})
sns.heatmap(avg_transition, cmap=sequential_green(), 
            xticklabels=ticks, yticklabels=ticks, cbar_kws={'label': 'No. of transitions'})
plt.tight_layout()
# plt.savefig('./figs/Fig4_ntransitions_modalitites.pdf')

# %%
lm = sns.lmplot(data=mod_df, x='NTr', y='value', scatter_kws={'s':0}, line_kws={'color':'Grey','alpha':0.5},
                ci=None)
lm.fig.set_dpi(250)
plt.rcParams.update({'font.size': 12})
pal_list = np.array(sequential_green(return_palette=True))[[1,3,4,7]]
# pal_list = sequential_green(return_palette=True)[1:3] + sequential_green(return_palette=True)[4,7]
ax = sns.scatterplot(data=mod_df, x='NTr', y='value', hue='transition_type', hue_order=order,
                     palette=pal_list)

plt.ylabel('Whole-brain TCE [a.u.]')
plt.xlabel('No. of transitions [log$_{10}$]')
r,dof,p,ci,power = pg.rm_corr(data=mod_df, y='NTr', x='value', subject='subject').values[0]
p_str = get_significance_string(p, type='text').upper()
anchor = (1,0.3) if dataset == 'hcp' else (0.4,0.35)
ax.legend(title='Transition type', bbox_to_anchor=anchor, frameon=False)
plt.title(f"$r_{{rm}}$ = {r:.3f}\n{p_str}", loc='right', y=0.8)
plt.tight_layout()

ax.set_ylim([lower_ylim, ax.get_ylim()[1]])
sns.despine()

if dataset == 'hcp':
    ticks = np.log10(np.linspace(100,1000,num=10))
    log_labels = ['100'] + ['' for tick in ticks[2:]] + ['1000']
    plt.xticks(ticks=ticks, labels=log_labels)
elif dataset == 'tum':
    ticks = np.log10(np.array([1, 10, 100]))
    labels = ['1', '10', '100']
    plt.xticks(ticks=ticks, labels=labels)


# %% ######################################
# plt.rcParams.update({'font.size': 12})
# pal_list = sequential_green(return_palette=True)[0:2] + sequential_green(return_palette=True)[4:6]
# lm = sns.jointplot(data=mod_df, x='NTr', y='value', hue='transition_type', hue_order=order, 
#                    palette=pal_list)
# lm.fig.set_dpi(250)
# plt.ylabel('Whole-brain TCE [a.u.]')
# plt.xlabel('Log$_{10}$(#Transitions)')
# r,dof,p,ci,power = pg.rm_corr(data=mod_df, y='NTr', x='value', subject='subject').values[0]
# p_str = get_significance_string(p, type='text')
# sns.move_legend(lm.ax_joint, 'upper left', title='Transition type', bbox_to_anchor=(1.05,1), frameon=False)
# plt.title(f"$r_{{rm}}$ = {r:.3f}\n{p_str}", loc='right', y=0.75)
# plt.tight_layout()

# lm.ax_joint.set_ylim([lower_ylim, lm.ax_joint.get_ylim()[1]])
# lm.ax_marg_y.set_ylim([lower_ylim, lm.ax_joint.get_ylim()[1]])
# lm.ax_marg_y.set_zorder(-1)
# lm.ax_marg_y.tick_params(left=False)
# lm.ax_marg_x.tick_params(bottom=False)
# sns.despine(trim=True)
# plt.tight_layout()

# plt.savefig('./figs/Fig4_modTCE_vs_ntransitions.pdf')
