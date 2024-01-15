# %%
import os, warnings
warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "1"
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import pingouin as pg
import matplotlib.pyplot as plt
from nilearn.input_data import NiftiLabelsMasker
from scipy.stats import pearsonr, ttest_rel, ttest_1samp
from neuromaps.images import  dlabel_to_gifti
from neuromaps.stats import compare_images
from neuromaps.parcellate import Parcellater
from neuromaps.nulls import alexander_bloch, burt2020

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

# %% 
# Control costs
control_df = pd.read_csv(output_dir + f'average-control-energy_subject-level_{atlas}.csv')

# Control costs at lower threshold
control_thr0_df = pd.read_csv(output_dir + f'average-control-energy_subject-level_{atlas}_thr0.csv')

# No-RSN files
tmp_dir = '/poolz2/ceballos/tmp/'
no_rsn_files = sorted(glob.glob(tmp_dir + f'no_rsn/*'))

# tSNR files
tsnr_files = sorted(glob.glob(tmp_dir + f'tsnr/*'))

r_tce_tsnr = []
p_tce_tsnr = []

r_tce_thr0_tsnr = []
p_tce_thr0_tsnr = []

r_no_rsn_tsnr = []
p_no_rsn_tsnr = []

for i,subject in enumerate(subjects):
    print(subject)
    tce = control_df[control_df['subject']==subject]['E_control'].values
    tce_thr0 = control_thr0_df[control_thr0_df['subject']==subject]['E_control'].values
    no_rsn = np.load(no_rsn_files[i])
    
    tsnr = np.load(tsnr_files[i])

    # compare tSNR to tce and no_rsn respectively
    # spin tsnr 1000 times
    nulls = alexander_bloch(tsnr, atlas='fsLR', density='32k', n_perm=1000, parcellation=atlas_fslr,
                            seed=1234)
    
    # compare to tce and no_rsn
    r1, p1 = compare_images(tsnr, tce, nulls=nulls, metric='spearmanr')
    r2, p2 = compare_images(tsnr, tce_thr0, nulls=nulls, metric='spearmanr')
    r3, p3 = compare_images(tsnr, no_rsn, nulls=nulls, metric='spearmanr')
    
    r_tce_tsnr.append(r1)
    p_tce_tsnr.append(p1)
    r_tce_thr0_tsnr.append(r2)
    p_tce_thr0_tsnr.append(p2)
    r_no_rsn_tsnr.append(r3)
    p_no_rsn_tsnr.append(p3)

# %% plot r_tce_tsnr and r_no_rsn_tsnr distributions in one histogram
fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=200)
plt.rcParams.update({'font.size': 14})

# Plot correlation distributions
sns.histplot(r_tce_tsnr, ax=axes[0], bins=60, label='RSN states')
sns.histplot(r_tce_thr0_tsnr, ax=axes[0], bins=60, label='RSN states (thr=0)')
sns.histplot(r_no_rsn_tsnr, ax=axes[0], bins=60, label='Raw BOLD states')

axes[0].set_xlabel(r"Spearman's $\rho$")
axes[0].set_ylabel('Frequency')
axes[0].set_title('Correlation between control costs and tSNR', fontsize=14)
axes[0].legend(frameon=False)

# Plot p-values distributions
sns.histplot(p_tce_tsnr, ax=axes[1], bins=60, label='RSN states')
sns.histplot(p_tce_thr0_tsnr, ax=axes[1], bins=60, label='RSN states (thr=0)')
sns.histplot(p_no_rsn_tsnr, ax=axes[1], bins=60, label='Raw BOLD states')
axes[1].axvline(0.05, color='k', linestyle='--', label='p=0.05')

axes[1].set_xlabel('P-value')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Significance of correlation between control costs and tSNR', fontsize=14)
axes[1].legend(frameon=False)

plt.tight_layout()

# %% average control_costs, control_costs_thr0 and no_rsn across subjects and compare to tSNR average
tce_avg = control_df.groupby('ROI').mean()['E_control'].values
tce_avg_thr0 = control_thr0_df.groupby('ROI').mean()['E_control'].values
no_rsn_avg = np.array([np.load(fn) for fn in no_rsn_files]).mean(axis=0)
tsnr_avg = np.array([np.load(fn) for fn in tsnr_files]).mean(axis=0)

nulls = burt2020(tsnr_avg, atlas='MNI152', density=den, n_perm=1000, parcellation=atlas_vol,
                 seed=1234, n_jobs=-1)

r1, p1 = compare_images(tsnr_avg, tce_avg, nulls=nulls, metric='spearmanr')
print(f'Average tSNR vs average control costs: r={r1:.3f}, p={p1:.3f}')

r2, p2 = compare_images(tsnr_avg, tce_avg_thr0, nulls=nulls, metric='spearmanr')
print(f'Average tSNR vs average control costs (thr=0): r={r2:.3f}, p={p2:.3f}')

r3, p3 = compare_images(tsnr_avg, no_rsn_avg, nulls=nulls, metric='spearmanr')
print(f'Average tSNR vs average no_rsn: r={r3:.3f}, p={p3:.3f}')

# %%
# Plot average tSNR and average control costs
plt.rcParams.update({'font.size': 14})
plt.figure(dpi=150)
sns.regplot(x=tsnr_avg, y=tce_avg)
plt.xlabel('tSNR [0-1]')
plt.ylabel('TCE [a.u.]')
plt.title(f"RSN-modelling\nrho={r1:.3f}, P$_{{SMASH}}$={p1:.3f}", y=1.03)

# Plot average tSNR and average control costs (thr=0)
plt.figure(dpi=150)
sns.regplot(x=tsnr_avg, y=tce_avg_thr0)
plt.xlabel('tSNR [0-1]')
plt.ylabel('TCE [a.u.]')
plt.title(f"RSN-modelling (thr=0)\nrho={r2:.3f}, P$_{{SMASH}}$={p2:.3f}", y=1.03)

# Plot average tSNR and average no_rsn
plt.figure(dpi=150)
sns.regplot(x=tsnr_avg, y=no_rsn_avg)
plt.xlabel('tSNR [0-1]')
plt.ylabel('TCE [a.u.]')
plt.title(f"Raw BOLD states\nrho={r3:.3f}, P$_{{SMASH}}$={p3:.3f}", y=1.03)

# # Plot average tSNR and average control costs (thr=0)
# ax = sns.jointplot(x=tsnr_avg, y=tce_avg_thr0, kind='reg')
# ax.set_axis_labels('tSNR [0-1]', 'TCE [a.u.]')
# ax.fig.suptitle(f"RSN-modelling (thr=0)\nrho={r2:.3f}, P$_{{SMASH}}$={p2:.3f}", y=1.08)
# ax.fig.set_dpi(150)

# # Plot average tSNR and average no_rsn
# ax = sns.jointplot(x=tsnr_avg, y=no_rsn_avg, kind='reg')
# ax.set_axis_labels('tSNR [0-1]', 'TCE [a.u.]')
# ax.fig.suptitle(f"Raw BOLD states\nrho={r3:.3f}, P$_{{SMASH}}$={p3:.3f}", y=1.08)
# ax.fig.set_dpi(150)
