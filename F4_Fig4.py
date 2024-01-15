# %% 
import warnings
warnings.filterwarnings("ignore")
import glob, os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from nilearn.input_data import NiftiLabelsMasker
from scipy.stats import spearmanr
from neuromaps.datasets import fetch_annotation
from neuromaps.images import dlabel_to_gifti, annot_to_gifti
from neuromaps.stats import compare_images
from neuromaps.parcellate import Parcellater
from neuromaps.nulls import burt2020
from matplotlib.gridspec import GridSpec
from src.cmaps import categorical_cmap

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
    surf_den = '32k'
    atlas_vol = root_dir + f'Gordon2016_space-MNI152_den-{den}.nii'
    labels = ['AUD', 'CoP', 'CoPar' ,'DMN', 'DAN', 'FrP', 'None', 'RT', 'SAL', 'SMH', 'SMM', 'VAN', 'VIS', 'NOTA']
    atlas_order = pd.read_pickle(root_dir + 'Gordon2016_333_LUT.pkl')
    atlas_fslr = dlabel_to_gifti(root_dir + 'Gordon2016_333_space-fsLR_den-32k.dlabel.nii')
    lab2mod={'AUD':'Unimodal','SMH':'Unimodal','SMM':'Unimodal','VIS':'Unimodal',
             'CoP':'Heteromodal','CoPar':'Heteromodal','DMN':'Heteromodal','FrP':'Heteromodal',
             'DAN':'Heteromodal','RT':'Heteromodal','SAL':'Heteromodal','VAN':'Heteromodal',
             'None':'None','NOTA':'Subthreshold'}
    
elif atlas == 'Schaefer400':
    den = '2mm'
    surf_den = '32k'
    atlas_vol = root_dir + f'Schaefer2018_400_7N_space-MNI152_den-{den}.nii.gz'
    labels = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default', 'NOTA']
    atlas_order = pd.read_pickle(root_dir + 'Schaefer2018_400_LUT.pkl')
    atlas_fslr = dlabel_to_gifti(root_dir + 'Schaefer2018_400_7N_space-fsLR_den-32k.dlabel.nii')
    lab2mod = {'Vis':'Unimodal', 'SomMot':'Unimodal', 'DorsAttn':'Heteromodal', 'SalVentAttn':'Heteromodal', 
               'Limbic':'Heteromodal', 'Cont':'Heteromodal', 'Default':'Heteromodal', 'NOTA':'Subthreshold'}
    
elif atlas == 'Schaefer200':
    den = '2mm'
    surf_den = '32k'
    atlas_vol = root_dir + f'Schaefer2018_200_7N_space-MNI152_den-{den}.nii.gz'
    labels = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default', 'NOTA']
    atlas_order = pd.read_pickle(root_dir + 'Schaefer2018_200_LUT.pkl')
    atlas_fslr = dlabel_to_gifti(root_dir + 'Schaefer2018_200_7N_space-fsLR_den-32k.dlabel.nii')
    lab2mod = {'Vis':'Unimodal', 'SomMot':'Unimodal', 'DorsAttn':'Heteromodal', 'SalVentAttn':'Heteromodal', 
               'Limbic':'Heteromodal', 'Cont':'Heteromodal', 'Default':'Heteromodal', 'NOTA':'Subthreshold'}

nrois = len(atlas_order)
nlabels = len(labels)
id2net=dict(zip(np.arange(nlabels)+1,labels))
mni = f'./data/MNI152_T1_{den}_brain.nii.gz'

# maskers to parcellate maps
mask = NiftiLabelsMasker(atlas_vol).fit(mni)

# %% Prepare average control energy for spatial map comparison
control_df = pd.read_csv(f'./results/{dataset}/average-control-energy_subject-level_{atlas}.csv')
avg_ctrl = control_df.groupby('ROI').mean()['E_control'].values

#%% Figure 4a
tum_cmrglc = './data/annotations/tum/cmrglc/MNI152/source-tum_desc-cmrglc_space-MN152_den-3mm.nii.gz'
roi_cmrglc = mask.fit_transform(tum_cmrglc)[0]
nulls_fn = f'./results/burt2020_cmrglc-nulls_{atlas}.npy'
if os.path.exists(nulls_fn):
    nulls = np.load(nulls_fn)
else:
    nulls = burt2020(roi_cmrglc, atlas='MNI152', density=den, n_perm=10000, parcellation=atlas_vol,
                    n_proc=-1, seed=1234)
    np.save(nulls_fn, nulls)

# Using Spearman correlation
glc_corr, p_glc = compare_images(roi_cmrglc, avg_ctrl, nulls=nulls, metric='spearmanr')
glc_corr_nulls = []
for ar in nulls.T:
    glc_corr_nulls.append(spearmanr(ar, roi_cmrglc.flatten())[0])
    
fig = plt.figure(dpi=250)
plt.rcParams.update({'font.size': 16})
lw = 5
gs = GridSpec(2, 1, height_ratios=[0.25, 1.75])
color = categorical_cmap(return_palette=True, n_colors=3)[0]
ace_color = categorical_cmap(return_palette=True, n_colors=3)[1]

ax1 = fig.add_subplot(gs[0])
ax1.scatter([glc_corr], [0], color=color, s=50)
sns.boxplot(x=glc_corr_nulls, width=0.5, ax=ax1, zorder=0,
            boxprops = {'facecolor':'none',
                        'edgecolor':'black'})
sns.despine(ax=ax1, left=True, right=True, top=True, bottom=True)
ax1.set(yticklabels=[], xticklabels=[], xlabel=None, ylabel=None)
ax1.tick_params(tick1On=False)

ax2 = fig.add_subplot(gs[1])
sns.histplot(glc_corr_nulls, stat='density', kde=True, alpha=0.1, color=ace_color, ax=ax2, line_kws={'lw':lw})
ax2.axvline(glc_corr, color=color, lw=lw)
ax2.set_xlabel(r"Spearman's $\rho$",fontsize='large')
bottom, top = plt.ylim()
plt.text(glc_corr+0.03,bottom+2,r'$\rho$'+f' =  {glc_corr:.3f}\n$p_{{SMASH}}$ = {p_glc:.3f}',fontsize='small')
plt.legend(['Spatial nulls','Observation'],bbox_to_anchor=(0.8,1),frameon=False)
sns.despine(ax=ax2)


plt.rcParams.update({'font.size': 16})
color = categorical_cmap(return_palette=True, n_colors=3)[0]
ax = sns.jointplot(x=roi_cmrglc, y=avg_ctrl, kind='reg', color=color, marginal_kws={'kde':True}, 
                   joint_kws={'scatter_kws':{'s':50}})
ax.ax_joint.set_xlabel('CMR$_{glc}$ [$\\frac{\mu mol}{100g*min}$]',size=20)
ax.ax_joint.set_ylabel('TCE [a.u.] ',size=20)
ax.ax_joint.set_xticks(np.arange(15, 45, 5))
ax.ax_joint.set_yticks(ax.ax_joint.get_yticks())
ax.fig.set_dpi(250)
sns.despine(ax=ax.ax_joint, offset=9, trim=True)

# %% Figure 4b
cmrox = fetch_annotation(source='raichle', desc='cmr02', space='fsLR', res='164k', data_dir=root_dir)
if atlas == 'Schaefer400' or atlas == 'Schaefer200':
    atlas_fs = annot_to_gifti((f'./data/Schaefer2018_{nrois}_7N_space-fsaverage_hemi-L.annot',
                               f'./data/Schaefer2018_{nrois}_7N_space-fsaverage_hemi-R.annot'))
    surf_masker = Parcellater(atlas_fs,'fsaverage')
else:
    surf_masker = Parcellater(atlas_fslr,'fslr')
roi_cmrox = surf_masker.fit_transform(cmrox,'fslr')

nulls_fn = f'./results/burt2020_cmro2-nulls_{atlas}.npy'
if os.path.exists(nulls_fn):
    nulls = np.load(nulls_fn)
else:
    nulls = burt2020(roi_cmrox, atlas='MNI152', density=den, n_perm=10000, parcellation=atlas_vol,
                    n_proc=-1, seed=1234)
    np.save(nulls_fn, nulls)   

# Using Spearman correlation
ox_corr, p_ox = compare_images(roi_cmrox, avg_ctrl, nulls=nulls, metric='spearmanr')
ox_corr_nulls = []
for ar in nulls.T:
    ox_corr_nulls.append(spearmanr(ar, roi_cmrox.flatten())[0])
    
fig = plt.figure(dpi=250)
plt.rcParams.update({'font.size': 16})
lw = 5
gs = GridSpec(2, 1, height_ratios=[0.25, 1.75])
color = categorical_cmap(return_palette=True, n_colors=3)[-1]
ace_color = categorical_cmap(return_palette=True, n_colors=3)[1]

ax1 = fig.add_subplot(gs[0])
ax1.scatter([ox_corr], [0], color=color, s=50)
sns.boxplot(x=ox_corr_nulls, width=0.5, ax=ax1, zorder=0,
            boxprops = {'facecolor':'none',
                        'edgecolor':'black'})
sns.despine(ax=ax1, left=True, right=True, top=True, bottom=True)
ax1.set(yticklabels=[], xticklabels=[], xlabel=None, ylabel=None)
ax1.tick_params(tick1On=False)

ax2 = fig.add_subplot(gs[1])
sns.histplot(ox_corr_nulls, stat='density', kde=True, alpha=0.1, color=ace_color, ax=ax2, line_kws={'lw':lw})
ax2.axvline(ox_corr, color=color, lw=lw)
ax2.set_xlabel(r"Spearman's $\rho$",fontsize='large')
bottom,top = plt.ylim()
plt.text(ox_corr+0.03,bottom+1,r'$\rho$'+f' =  {ox_corr:.3f}\n$p_{{SMASH}}$ = {p_ox:.3f}',fontsize='small')
plt.legend(['Spatial nulls','Observation'],bbox_to_anchor=(0.9,1),frameon=False)
sns.despine(ax=ax2)

plt.rcParams.update({'font.size': 16})
color = categorical_cmap(return_palette=True, n_colors=3)[-1]
ax = sns.jointplot(x=roi_cmrox, y=avg_ctrl, kind='reg', color=color, marginal_kws={'kde':True}, 
                   joint_kws={'scatter_kws':{'s':50}})
ax.ax_joint.set_xlabel('CMR$_{O2}$ [ $\\frac{ Bq }{ mL }$ ]',size=20)
ax.ax_joint.set_ylabel('TCE [a.u.]',size=20)
ax.ax_joint.set_xticks(np.arange(3500, 7000, 1000))
ax.ax_joint.set_yticks(ax.ax_joint.get_yticks())
ax.fig.set_dpi(250)
sns.despine(ax=ax.ax_joint, offset=9, trim=True)
