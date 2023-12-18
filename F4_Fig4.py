# %% 
import warnings
warnings.filterwarnings("ignore")
import glob, os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
import seaborn as sns
import scipy as sp
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
from src.utils import test_multigroup_mean
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
    labels = ['AUD', 'CoP', 'CoPar' ,'DMN', 'DAN', 'FrP', 'None', 'RT', 'SAL','SMH','SMM','VAN','VIS','NOTA']
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

#%% CMRglc from TUM data
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
plt.legend(['Nulls','Observation'],bbox_to_anchor=(0.8,1),frameon=False)
sns.despine(ax=ax2)
# plt.savefig('./figs/Fig5_cmrglc_spin_nulls.pdf')

plt.rcParams.update({'font.size': 16})
color = categorical_cmap(return_palette=True, n_colors=3)[0]
ax = sns.jointplot(x=roi_cmrglc, y=avg_ctrl, kind='reg', color=color, marginal_kws={'kde':True}, 
                   joint_kws={'scatter_kws':{'s':50}})
ax.ax_joint.set_xlabel('CMR$_{glc}$ [$\\frac{\mu mol}{100g*min}$]',size=15)
ax.ax_joint.set_ylabel('TCE [a.u.] ',size=15)
ax.ax_joint.set_xticks(np.arange(15, 45, 5))
ax.ax_joint.set_yticks(ax.ax_joint.get_yticks())
ax.fig.set_dpi(250)
sns.despine(ax=ax.ax_joint, offset=9, trim=True)
# plt.savefig('./figs/Fig5_cmrglc_jointplot.pdf')

# %% CMRO2 from neuromaps
cmrox = fetch_annotation(source='raichle', desc='cmr02', space='fsLR', res='164k', data_dir=root_dir)
if atlas == 'Schaefer400' or atlas == 'Schaefer200':
    atlas_fs = annot_to_gifti((f'./data/Schaefer2018_{nrois}_7N_space-fsaverage_hemi-L.annot',
                               f'./data/Schaefer2018_{nrois}_7N_space-fsaverage_hemi-R.annot'))
    surf_masker = Parcellater(atlas_fs,'fsaverage')
else:
    roi_cmrox = surf_masker.fit_transform(cmrox,'fslr')
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
plt.legend(['Nulls','Observation'],bbox_to_anchor=(0.9,1),frameon=False)
sns.despine(ax=ax2)
# plt.savefig('./figs/Fig5_cmro2_spin_nulls.pdf')

plt.rcParams.update({'font.size': 16})
color = categorical_cmap(return_palette=True, n_colors=3)[-1]
ax = sns.jointplot(x=roi_cmrox, y=avg_ctrl, kind='reg', color=color, marginal_kws={'kde':True}, 
                   joint_kws={'scatter_kws':{'s':50}})
ax.ax_joint.set_xlabel('CMR$_{O2}$ [ $\\frac{ Bq }{ mL }$ ]',size=15)
ax.ax_joint.set_ylabel('TCE [a.u.]',size=15)
ax.ax_joint.set_xticks(np.arange(3500, 7000, 1000))
ax.ax_joint.set_yticks(ax.ax_joint.get_yticks())
ax.fig.set_dpi(250)
sns.despine(ax=ax.ax_joint, offset=9, trim=True)
# plt.savefig('./figs/Fig5_cmro2_jointplot.pdf')

# %% Compare CMRglc and CMRO2 to degree
plt.rcParams.update({'font.size': 16})
color = categorical_cmap(return_palette=True, n_colors=3)[-1]

if dataset == 'monash':
    deg = np.sum(A, axis=-1)
    # To-Do: test what the ylim is for the group average connectome
else:
    allA = np.array([np.load(fn) \
            for fn in sorted(glob.glob(dataset_dir + f'*streamline*{atlas}.npy'))])
    deg = np.sum(allA.mean(axis=0), axis=-1)
    
    # CMRglc
    ax = sns.jointplot(x=roi_cmrglc, y=deg, kind='hex', color=color, marginal_kws={'kde':True}, joint_kws={'gridsize':25})
    ax.ax_joint.set_xlim([15, 40])
    ax.ax_joint.set_ylim([0, 140000])
    # TO-DO: implement scientific axis format
    # ax.ax_joint.ticklabel_format(axis='y', style='scientific')
    ax.fig.set_dpi(250)
    sns.despine(ax=ax.ax_joint, offset=9) 
    
    # CMRO2
    ax = sns.jointplot(x=roi_cmrox, y=deg, kind='hex', color=color, marginal_kws={'kde':True}, joint_kws={'gridsize':25})
    ax.ax_joint.set_ylim([0, 140000])
    ax.ax_joint.set_xlim([2000, 7000])
    ax.fig.set_dpi(250)
    sns.despine(ax=ax.ax_joint, offset=9)


#######################################################
#                         OLD
#######################################################
# %% 
# #### Compare energy rank between ACE, CMRglc and CMRO2
plot_df = pd.DataFrame(np.vstack((roi_cmrglc, avg_ctrl, roi_cmrox)).T,columns=['CMRglc', 'E_control', 'CMRO2'])
plot_df = plot_df.rank()
plot_df['ROI'] = atlas_order['ROI']
plot_df['network'] = atlas_order['network']
plot_df['mod'] = plot_df['network'].map(lab2mod)
plot_df = pd.melt(plot_df,id_vars=['ROI','network','mod'],value_vars=['E_control','CMRglc','CMRO2'],var_name='energy')

plt.figure(figsize=(4,10),dpi=250)
color = categorical_cmap(return_palette=True, n_colors=3)
ax = sns.barplot(data=plot_df,y='mod',x='value',hue='energy',ci='sd', alpha=0.8, estimator=np.median, palette=color)
sns.despine(offset={'bottom':5})
ax.legend(title='Energy Type', handles=ax.get_legend_handles_labels()[0], labels=['CMR$_{glc}$', 'ACE', 'CMR$_{O2}$'], 
          frameon=False, bbox_to_anchor=(1.1, 1.02))
ax.set_xlabel('Rank',size=16)
ax.set_ylabel('')
plt.yticks(rotation=90)
for tick in ax.get_yticklabels():
    tick.set_verticalalignment("center")

print(test_multigroup_mean(plot_df[plot_df['mod']=='Heteromodal'], dv='value', group='energy'))
print(test_multigroup_mean(plot_df[plot_df['mod']=='Unimodal'], dv='value', group='energy'))

# %% 
# #### Compare energy rank between ACE, CMRglc and CMRO2
plot_df = pd.DataFrame(np.vstack((roi_cmrglc, avg_ctrl, roi_cmrox)).T,columns=['CMRglc', 'E_control', 'CMRO2'])
plot_df = plot_df.rank()
plot_df['ROI'] = atlas_order['ROI']
plot_df['network'] = atlas_order['network']
plot_df['mod'] = plot_df['network'].map(lab2mod)
plot_df = pd.melt(plot_df,id_vars=['ROI','network','mod'],value_vars=['E_control','CMRglc','CMRO2'],var_name='energy')

plt.figure(figsize=(10,4),dpi=250)
color = categorical_cmap(return_palette=True, n_colors=3)
ax = sns.barplot(data=plot_df,x='mod',y='value',hue='energy',ci='sd', estimator=np.median, palette=color, alpha=0.9)
ax.legend(title='Energy Type', handles=ax.get_legend_handles_labels()[0], labels=['CMR$_{glc}$', 'ACE', 'CMR$_{O2}$'], 
          frameon=False, bbox_to_anchor=(1.01, 1.1))
ax.set_ylabel('Rank',size=16)
ax.set_ylim([0, nrois])
ax.set_xlabel('')
sns.despine(offset={'left':10})
# plt.xticks(rotation=90)
# for tick in ax.get_yticklabels():
#     tick.set_verticalalignment("center")

print(test_multigroup_mean(plot_df[plot_df['mod']=='Heteromodal'], dv='value', group='energy'))
print(test_multigroup_mean(plot_df[plot_df['mod']=='Unimodal'], dv='value', group='energy'))

# %%
# Compare results to statistical nulls
pr_cov_nulls = np.load('./results/PR-cov-null_average-control-energy_Gordon_333.npy')
pr_nocov_nulls = np.load('./results/PR-nocov-null_average-control-energy_Gordon_333.npy')
sc_str_nulls = np.load('./results/SC-str-null_average-control-energy_Gordon_333.npy')
sc_geo_nulls = np.load('./results/SC-geo-null_average-control-energy_Gordon_333.npy')

pr_cov_glc_corr = []
pr_nocov_glc_corr = []
sc_str_glc_corr = []
sc_geo_glc_corr = []

pr_cov_ox_corr = []
pr_nocov_ox_corr = []
sc_str_ox_corr = []
sc_geo_ox_corr = []

for n1,n2,n3,n4 in zip(pr_cov_nulls.T,pr_nocov_nulls.T,sc_str_nulls.T,sc_geo_nulls.T):
    pr_cov_glc_corr.append(spearmanr(n1, roi_cmrglc.flatten())[0])
    pr_cov_ox_corr.append(spearmanr(n1, roi_cmrox.flatten())[0])
    pr_nocov_glc_corr.append(spearmanr(n2, roi_cmrglc.flatten())[0])
    pr_nocov_ox_corr.append(spearmanr(n2, roi_cmrox.flatten())[0])
    
    sc_str_glc_corr.append(spearmanr(n3, roi_cmrglc.flatten())[0])
    sc_str_ox_corr.append(spearmanr(n3, roi_cmrox.flatten())[0])
    sc_geo_glc_corr.append(spearmanr(n4, roi_cmrglc.flatten())[0])
    sc_geo_ox_corr.append(spearmanr(n4, roi_cmrox.flatten())[0])

glc = [pr_cov_glc_corr,
       pr_nocov_glc_corr,
       sc_str_glc_corr,
       sc_geo_glc_corr]

ox = [pr_cov_ox_corr,
      pr_nocov_ox_corr,
      sc_str_ox_corr,
      sc_geo_ox_corr]
# %%
fig = plt.figure(figsize=(11, 4), dpi=200)
gs = GridSpec(2, 3, width_ratios=[2, 3, 3])

# Create the subplots
ax1 = GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[:, 1])
ax2 = GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[:, 2])

for i in range(4):
        tmp_ax1 = fig.add_subplot(ax1[i])
        sns.boxplot(x=glc[i], ax=tmp_ax1, width=0.5, whis=3, showmeans=True,
                    boxprops = {'facecolor':'none',
                                'edgecolor':'black'},
                    meanprops={'marker': '|',
                               'markeredgecolor': 'black',
                               'markersize': '25'})
        tmp_ax1.scatter([glc_corr], [0], color=sns.color_palette("Set2")[1], s=10)
        tmp_ax1.set_xlim([-0.05, 0.15])
        
        tmp_ax2 = fig.add_subplot(ax2[i])
        sns.boxplot(x=ox[i], ax=tmp_ax2, width=0.5, whis=3, showmeans=True,
                    boxprops = {'facecolor':'none',
                                'edgecolor':'black'},
                    meanprops={'marker': '|',
                               'markeredgecolor': 'black',
                               'markersize': '25'})
        tmp_ax2.scatter([ox_corr], [0], color=sns.color_palette("Set2")[2], s=10)
        tmp_ax2.set_xlim([-0.05, 0.45])
        
        if i < 3:
            sns.despine(ax=tmp_ax1, left=True, right=True, top=True, bottom=True)
            sns.despine(ax=tmp_ax2, left=True, right=True, top=True, bottom=True)
            tmp_ax1.set_xticks([])
            tmp_ax1.set_yticks([])
            tmp_ax1.set_xticklabels([])
            tmp_ax1.set_yticklabels([])
            
            tmp_ax2.set_xticks([])
            tmp_ax2.set_yticks([])
            tmp_ax2.set_xticklabels([])
            tmp_ax2.set_yticklabels([])
        else:
            sns.despine(ax=tmp_ax1, left=True, right=True, top=True)
            sns.despine(ax=tmp_ax2, left=True, right=True, top=True)
            
            tmp_ax1.set_yticks([])
            tmp_ax1.set_yticklabels([])
            tmp_ax1.set_xlabel(r"Spearman's $\rho$")
            
            tmp_ax2.set_yticks([])
            tmp_ax2.set_yticklabels([])
            tmp_ax2.set_xlabel(r"Spearman's $\rho$")