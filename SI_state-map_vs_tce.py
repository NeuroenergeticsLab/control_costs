# %%
import warnings,glob,time,itertools,os
warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from nilearn.input_data import NiftiLabelsMasker
from neuromaps.parcellate import Parcellater
from neuromaps.images import dlabel_to_gifti
from neuromaps.stats import compare_images
from neuromaps.nulls import burt2020, alexander_bloch
from scipy.stats import spearmanr
from src.cmaps import categorical_cmap
from joblib import Parallel,delayed
from src.plot_utils import custom_surf_plot
from src.cmaps import custom_coolwarm, sequential_green, sequential_blue

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
elif dataset == 'monash':
    subjects = [f'sub-{sid:02}' for sid in np.arange(1,28)]

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
roi2netid = dict(zip(atlas_order['ROI'],atlas_order['network_id']))
mni = f'./data/MNI152_T1_{den}_brain.nii.gz'

# maskers to parcellate maps
mask = NiftiLabelsMasker(atlas_vol).fit(mni)
surf_masker = Parcellater(atlas_fslr,'fslr')

# %% plot state counts as categorical plot for each label
state_counts = np.zeros((nlabels-1, len(subjects)))
for i,subj_id in enumerate(subjects):
    print(f'Starting with subject {subj_id}')
    targets = np.load(dataset_dir + f'{subj_id}_task-rest_bold_desc-time_labels_{atlas}.npy')
    for j,state in enumerate(np.unique(targets)[:-1]):
        state_counts[j,i] = np.count_nonzero(targets == state)
state_counts = state_counts.T

# plot barplot of each state count
new_labels = ['VIS', 'SMN', 'DAN', 'SAL', 'LIM', 'FPN', 'DMN']
fig, ax = plt.subplots(figsize=(6,4), dpi=150)
sns.barplot(data=pd.DataFrame(state_counts,columns=new_labels),ax=ax, errorbar='sd')
ax.set(xlabel='State', ylabel='Frequency')
sns.despine()
plt.tight_layout()

# %% multiply state coefficients by number of occurrences of each state
state_maps = pd.read_csv(output_dir + f'state-maps_subject-level_{atlas}.csv')
group_coeffs = np.zeros((len(subjects), nrois))

for i,subj_id in enumerate(subjects):
    print(f'Starting with subject {subj_id}')
    targets = np.load(dataset_dir + f'{subj_id}_task-rest_bold_desc-time_labels_{atlas}.npy')
    state_counts = [np.count_nonzero(targets == state) for state in np.arange(nlabels-1)]
    ntpoints = sum(state_counts)
    subj_df = state_maps[state_maps['subject']==subj_id]
    coeffs = np.zeros(nrois)
    for j,state in enumerate(labels[:-1]):
        state_coeffs = subj_df[subj_df['state'] == state]['value'].values
        coeffs += state_coeffs * (state_counts[j] / ntpoints)
    group_coeffs[i,:] = coeffs
    
avg_coeffs = group_coeffs.mean(axis=0)

# %% Compare to TCE
control_df = pd.read_csv(f'./results/{dataset}/average-control-energy_subject-level_{atlas}.csv')
avg_ctrl = control_df.groupby('ROI').mean()['E_control'].values

nulls = burt2020(avg_ctrl, atlas='MNI152', density='3mm', n_perm=1000, parcellation=atlas_vol,
                 seed=1234)

r, p = compare_images(avg_ctrl, avg_coeffs, nulls=nulls, metric='spearmanr')
coeff_corr_nulls = []
for ar in nulls.T:
    coeff_corr_nulls.append(spearmanr(ar, avg_coeffs)[0])

# %% plot statistical significance
plt.rcParams.update({'font.size': 14})

ax = sns.jointplot(x=avg_coeffs, y=avg_ctrl, kind='reg', marginal_kws={'kde':True})
ax.set_axis_labels('State coefficients [a.u.]', 'Control energy [a.u.]')
ax.fig.set_dpi(150)

fig = plt.figure(figsize=(6, 4), dpi=150)
gs = GridSpec(3, 1, height_ratios=[1, 0.5, 4])
color = 'blue'

ax1 = fig.add_subplot(gs[0:1])
ax1.scatter([r], [0], color=color, s=20)
sns.boxplot(x=coeff_corr_nulls, width=0.5, ax=ax1, zorder=0,
            boxprops = {'facecolor':'none',
                        'edgecolor':'black'})
sns.despine(ax=ax1, left=True, right=True, top=True, bottom=True)
ax1.set(yticklabels=[], xticklabels=[], xlabel=None, ylabel=None)
ax1.tick_params(tick1On=False)

ax2 = fig.add_subplot(gs[2:3])
sns.histplot(coeff_corr_nulls, stat='density', kde=True, alpha=0.1, ax=ax2)
ax2.axvline(r, color=color)
ax2.set_xlabel(r"Spearman's $\rho$",fontsize='large')
bottom,top = plt.ylim()
plt.text(r+0.02,bottom+1.5,r'$\rho$'+f' =  {r:.3f}\n$P_{{SMASH}}$ = {p:.3f}',fontsize='small')
plt.legend(['Spin nulls','Observation'],bbox_to_anchor=(1.05,1),frameon=False)
sns.despine(ax=ax2)
