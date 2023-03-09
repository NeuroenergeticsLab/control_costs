#####################################################################################################################################
# %% Script to show statistics and plots using the HCP dataset
#####################################################################################################################################
import warnings
warnings.filterwarnings("ignore")
import glob
import numpy as np
import nibabel as nib
import pandas as pd
import seaborn as sns
import pingouin as pg
import matplotlib.pyplot as plt
from scipy.io import loadmat
from nilearn.input_data import NiftiLabelsMasker
from scipy.stats import pearsonr,ttest_rel,ttest_1samp,kruskal,zscore
from network_control.utils import matrix_normalization
from neuromaps.datasets import fetch_annotation
from neuromaps.images import relabel_gifti
from neuromaps.stats import compare_images
from neuromaps.parcellate import Parcellater
from neuromaps.transforms import fslr_to_fslr
from neuromaps.nulls import burt2020


# %% General variables

dataset = 'hcp'
root_dir = './data/'
dataset_dir = root_dir + f'{dataset}/'
output_dir = f'./results/{dataset}/'

if dataset == 'hcp':
    subjects = sorted([fn.split('/')[-1][:10] for fn in glob.glob(dataset_dir + 'sub*labels.npy')])
elif dataset == 'monash':
    subjects = [f'sub-{sid:02}' for sid in np.arange(1,28)]

atlas = 'Gordon'
if atlas == 'Gordon':
    den = '3mm'
    labels = ['AUD', 'CoP', 'CoPar' ,'DMN', 'DAN', 'FrP', 'None', 'RT', 'SAL','SMH','SMM','VAN','VIS','NOTA']
    nlabels = len(labels)
    atlas_vol = root_dir + f'Gordon2016_space-MNI152_den-{den}.nii'
    atlas_order = pd.read_csv(root_dir + 'gordon2016_parcels.csv')
    atlas_order = atlas_order.rename(columns={'ParcelID':'ROI'})
    atlas_order = atlas_order.drop(columns=[atlas_order.columns[-1],'Surface area (mm2)','Centroid (MNI)'])
    atl2id = dict(zip(labels,np.arange(nlabels)+1))
    atl_id2label = dict(zip(atlas_order['ROI'].tolist(), atlas_order['Community']))
    atlas_order['CommunityID'] = atlas_order['Community'].map(atl2id)
    atlas_fslr = (root_dir + 'Gordon2016_space-fsLR_den-32k_hemi-L.func.gii',
                  root_dir + 'Gordon2016_space-fsLR_den-32k_hemi-R.func.gii')
    atlas_fslr = relabel_gifti(atlas_fslr,background=0)
    A = loadmat('./data/Gordon2016_whole-brain_SC.mat')['connectivity'] 
    lab2mod={'AUD':'Unimodal','SMH':'Unimodal','SMM':'Unimodal','VIS':'Unimodal',
             'CoP':'Multimodal','CoPar':'Multimodal','DMN':'Multimodal','FrP':'Multimodal',
             'DAN':'Multimodal','RT':'Multimodal','SAL':'Multimodal','VAN':'Multimodal',
             'None':'None','NOTA':'Subthreshold'}
    
elif atlas == 'Schaefer':
    den = '2mm'
    labels = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default', 'NOTA']
    nlabels = len(labels)
    atlas_vol = root_dir + f'Schaefer2018_400_7N_space-MNI152_den-{den}.nii.gz'
    atlas_order = pd.read_csv(root_dir + 'Schaefer2018_400_7N_order.txt',sep='\t',header=None)
    atlas_order = atlas_order.rename(columns={0:'ROI'})
    atlas_order['network'] = atlas_order[1].str.split('_').str.get(2)
    atl2id = dict(zip(atlas_order['network'].unique(),np.arange(nlabels)+1))
    atl_id2label = dict(zip(atlas_order['ROI'].tolist(), atlas_order['network']))
    atlas_order['network_id'] = atlas_order['network'].map(atl2id)
    atlas_fslr = (root_dir + 'Schaefer2018_400_7N_space-fsLR_den-32k_hemi-L.func.gii',
                  root_dir + 'Schaefer2018_400_7N_space-fsLR_den-32k_hemi-R.func.gii')
    atlas_fslr = relabel_gifti(atlas_fslr,background=0)
    A = loadmat('./data/Schaefer2018_400_7N_whole-brain_SC.mat')['connectivity']
    lab2mod = {'Vis':'Unimodal', 'SomMot':'Unimodal', 'DorsAttn':'Multimodal', 'SalVentAttn':'Multimodal', 
               'Limbic':'Multimodal', 'Cont':'Multimodal', 'Default':'Multimodal', 'NOTA':'Subthreshold'}

nrois = len(atlas_order)
id2atl=dict(zip(np.arange(nlabels)+1,labels))
mni = f'./data/MNI152_T1_{den}_brain.nii.gz'
mask = NiftiLabelsMasker(atlas_vol).fit(mni)

# CMRglc map
tum_cmrglc = './data/avg_cmrglc_subjs-20_45min_mcf_fwhm-6_quant-cmrglc_acq-2242min_pvc-pveseg_mni-3mm.nii.gz'
roi_cmrglc = mask.fit_transform(tum_cmrglc)[0]

# CMRO2 map
cmrox_164k = fetch_annotation(source='raichle', desc='cmr02', space='fsLR', res='164k', data_dir=root_dir)
#### cmrox = fslr_to_fslr(cmrox_164k,'32k')
surf_masker = Parcellater(atlas_fslr,'fslr')
#### roi_cmrox = surf_masker.fit_transform(cmrox,'fslr')

# %%
cmrox = ('/home/tumnic/eceballos/neuromaps-data/annotations/raichle/cmr02/fsLR/source-raichle_desc-cmr02_space-fsLR_den-32k_hemi-L_feature.func.gii.gz',
         '/home/tumnic/eceballos/neuromaps-data/annotations/raichle/cmr02/fsLR/source-raichle_desc-cmr02_space-fsLR_den-32k_hemi-R_feature.func.gii.gz')
roi_cmrox = surf_masker.fit_transform(cmrox,'fslr')

# %% [markdown]
# ### Compare spatial maps

# %%
# control_df = pd.read_csv(f'./results/{dataset}/average-optimal-control-energy_roiwise_subject-level_Bmap_{atlas}_{nrois}.csv',index_col=1)
control_df = pd.read_csv('./results/hcp_nomapper/average-optimal-control-energy_roiwise_subject-level_Bmap_Gordon_333.csv',index_col=1)
avg_ctrl = control_df.groupby('ROI').mean()['E_control'].values
nulls = burt2020(avg_ctrl,atlas='MNI152',density=den,n_perm=1000,parcellation=atlas_vol,seed=0)

# %%
def stat_annot(y_pos, x_pos, w, color, lw):
    y1, y2 = y_pos-0.4, y_pos+0.4
    plt.plot([x_pos, x_pos+w, x_pos+w, x_pos], [y1, y1, y2, y2], lw=lw, c=color,clip_on=False)
    plt.text(x_pos+0.05, y_pos+0.1, "ns", ha='center', va='bottom', color=color,fontsize=7);

plot_df = pd.DataFrame(np.vstack((avg_ctrl,roi_cmrglc,roi_cmrox)).T,columns=['E_control','CMRglc','CMRO2'])
plot_df = (plot_df-plot_df.min())/(plot_df.max()-plot_df.min())
plot_df['ROI'] = np.arange(nrois).astype(int)+1
plot_df['network'] = plot_df['ROI'].map(atl_id2label)
plot_df = plot_df[plot_df['network']!='None']
ns = []
for i,lab in enumerate(plot_df['network'].unique()):
    tmp = plot_df[plot_df['network']==lab].iloc[:,:3].values
    H,pval = kruskal(tmp[:,0],tmp[:,1],tmp[:,2])
    if pval>=0.001:
        ns.append(i)
plot_df['mod'] = plot_df['network'].map(lab2mod)
plot_df = pd.melt(plot_df,id_vars=['ROI','network','mod'],value_vars=['E_control','CMRglc','CMRO2'],var_name='energy')

plt.figure(figsize=(5,8),dpi=100)
ax = sns.barplot(data=plot_df,y='network',x='value',hue='energy',ci='sd',palette=sns.color_palette(), estimator=np.median)
sns.despine(offset={'bottom':5})
ax.legend(title='Energy Type', handles=ax.get_legend_handles_labels()[0], labels=['ACE','CMR$_{glc}$','CMR$_{O2}$'], 
          frameon=False, bbox_to_anchor=(1.3, 1.02))
ax.set_xlim(0,1)
ax.set_xlabel('Normalized Energy Consumption',size=13)
ax.set_ylabel('Network',size=16)
plt.tight_layout()

[stat_annot(pos, 1.03, 0.01, 'k', 0.8) for pos in ns];

# %%
glc_corr, p_glc = compare_images(avg_ctrl,roi_cmrglc,nulls=nulls)

glc_corr_nulls = []
for ar in nulls.T:
    glc_corr_nulls.append(pearsonr(ar,roi_cmrglc)[0])
    
plt.figure(dpi=250)
ax = sns.histplot(glc_corr_nulls,stat='density',kde=True,alpha=0.4)
ax.axvline(glc_corr,color=sns.color_palette()[1])
ax.set_xlabel("Pearson's $r$",fontsize='large')
bottom,top = plt.ylim()
plt.text(glc_corr+0.02,bottom+1,f'$r$  =  {glc_corr:.3f}\n$P_{{spin}}$ = {p_glc:.3f}',fontsize='small')
plt.legend(['Null data','Observation'],bbox_to_anchor=(0.8,1),frameon=False)
sns.despine()
plt.title('ACE ↔ $CMR_{glc}$')

ax = sns.displot(x=avg_ctrl,y=roi_cmrglc, bins=(20, 20), cbar=True, cbar_kws={'label':'count', 'format':'%0.f','ticks':np.linspace(0,20,num=5)},
                 color=sns.color_palette("YlOrBr")[-3]);
ax.ax.set_title(f'$r$  =  {glc_corr:.3f}\n$P_{{spin}}$ = {p_glc:.3f}',x=0.85,y=0.1)
ax.ax.set_ylabel('CMR$_{glc}$ [$\\frac{\mu mol}{100g*min}$]',size=15)
ax.ax.set_xlabel('ACE [a.u.] ',size=15)
ax.fig.set_figwidth(6)
ax.fig.set_dpi(150)

# %%
ox_corr, p_ox = compare_images(avg_ctrl,roi_cmrox,nulls=nulls)
ox_corr_nulls = []
for ar in nulls.T:
    ox_corr_nulls.append(pearsonr(ar,roi_cmrox)[0])

plt.figure(dpi=250)
ax = sns.histplot(ox_corr_nulls,stat='density',kde=True,alpha=0.4)
ax.axvline(ox_corr,color=sns.color_palette()[2])
ax.set_xlabel("Pearson's $r$",fontsize='large')
bottom,top = plt.ylim()
plt.text(ox_corr+0.02,bottom+1,f'$r$  =  {ox_corr:.3f}\n$P_{{spin}}$ = {p_ox:.3f}',fontsize='small')
plt.legend(['Null data','Observation'],bbox_to_anchor=(0.8,1),frameon=False)
sns.despine()
plt.title('ACE ↔ $CMR_{O2}$')

ax = sns.displot(x=avg_ctrl,y=zscore(roi_cmrox), bins=(20, 20), cbar=True, cbar_kws={'label':'count', 'format':'%0.f','ticks':np.linspace(0,20,num=5)},
                 color=sns.light_palette("seagreen")[-1]);
ax.ax.set_title(f'$r$  =  {ox_corr:.3f}\n$P_{{spin}}$ = {p_ox:.3f}',x=0.85,y=0.1)
ax.ax.set_ylabel('Z-score (CMR$_{O2}$) [a.u.]',size=15)
ax.ax.set_xlabel('ACE [a.u.]',size=15)
ax.fig.set_figwidth(6)
ax.fig.set_dpi(150)

# %% [markdown]
# ### 

# %% Relationship to structural connectome
# ### 

# %%
A_norm = matrix_normalization(A,c=1,version='continuous').astype(np.float16)
np.fill_diagonal(A_norm, 0)
strength = A_norm.sum(axis=0)
corr, pval = compare_images(avg_ctrl,strength,nulls=nulls)
ax = sns.displot(x=avg_ctrl,y=strength, bins=(20, 20), cbar=True, cbar_kws={'label':'count', 'format':'%0.f','ticks':np.linspace(0,20,num=5)});
ax.ax.set_title(f'$r$  =  {corr:.3f}\n$P_{{spin}}$ = {pval:.3f}',x=0.85,y=0.9)
ax.ax.set_ylabel('SC$_{strength}$ [a.u]',size=15)
ax.ax.set_xlabel('ACE [a.u.]',size=15)
ax.fig.set_figwidth(6)
ax.fig.set_dpi(150)

# %% Relationship to synaptic density

# %%
expression = pd.read_csv('./data/AHBA_gene-expression_interpolated_Gordon_333.csv')
sv2a = expression['SV2A'].values

corr, pval = compare_images(avg_ctrl,sv2a,nulls=nulls)

plot_df = pd.DataFrame(np.vstack((avg_ctrl,sv2a)).T,columns=['ACE','SV2A'])

# lm = sns.lmplot(data=plot_df, x='ACE', y='SV2A',scatter=False,line_kws={'color':sns.color_palette("crest")[-1],'alpha':0.6})
ax = sns.displot(data=plot_df, x='ACE', y='SV2A', bins=(20, 20), cbar=True, cbar_kws={'label':'count', 'format':'%0.f','ticks':np.linspace(0,20,num=5)},alpha=0.95,zorder=-1,
                  color=sns.color_palette("dark:salmon_r")[1])
ax.ax.set_title(f'$r$  =  {corr:.3f}\n$P_{{spin}}$ = {pval:.3f}',x=0.85,y=0.1)
ax.ax.set_ylabel('SV2A gene expression [a.u]',size=15)
ax.ax.set_xlabel('ACE [a.u.]',size=15);
ax.fig.set_figwidth(6)
ax.fig.set_dpi(150)

# %% [markdown]
# ### Average control energy grouped by modality

# %%
##### mod_df = pd.read_csv(f'./results/{dataset}/average-optimal-control-energy_modality_roiwise_subject-level_Bmap_{atlas}_{nrois}.csv').iloc[:,-6]
mod_df = pd.read_csv('./results/hcp_nomapper/average-optimal-control-energy_modality_roiwise_subject-level_Bmap_Gordon_333.csv').iloc[:,-6:]

# rename transition names for readability
trans_dict = {'E_mm':'M→M','E_mu':'M→U','E_um':'U→M','E_uu':'U→U'}
mod_df = mod_df.rename(columns=trans_dict)

# order by median energy per transition type
col_idx = np.mean(mod_df.values[:,:4],axis=0).argsort()
order = mod_df.columns[col_idx].to_list()

# average by subject for each modality
mod_df = mod_df.groupby('subject').mean().iloc[:,:-1].reset_index()

# transform to long format for plotting
mod_df = pd.melt(mod_df,value_vars=['U→U','M→U','U→M','M→M'],id_vars='subject',var_name='transition_type')
mod_df = mod_df.sort_values(['subject','transition_type'],ignore_index=True)

# %%
palette = 'Spectral'
pal_list = sns.color_palette(f"{palette}")[:2] + sns.color_palette(f"{palette}")[-2:]
plt.figure(dpi=250)
with sns.axes_style(style="whitegrid"):
    ax = sns.boxplot(data=mod_df,x='transition_type',y='value',width=0.3,linewidth=1.5,order=order,palette=pal_list)
    sns.stripplot(data=mod_df,x='transition_type',y='value',alpha=0.1,ax=ax,linewidth=1,order=order,palette=pal_list)
    sns.despine(ax=ax,bottom=True)
    # ax.set_title('HCP')
    ax.tick_params(tick1On=False)
    plt.xlabel("Transition type")
    plt.ylabel("Whole-brain ACE [a.u.]")
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    # plt.savefig('./graphs/modality_transition_cost.pdf')

# %%
# T-Test between intra and intermodal transitions
intra = mod_df[(mod_df.transition_type=='M→M')|(mod_df.transition_type=='U→U')]['value'].values
inter = mod_df[(mod_df.transition_type=='U→M')|(mod_df.transition_type=='M→U')]['value'].values
ttest_rel(inter, intra)

# %% [markdown]
# ### Contrast average control energy to number of transitions

# %%
# Number of transitions
all_mtx = np.array([])

for subj_id in subjects:
    targets = np.load(dataset_dir + f'{subj_id}_task-rest_bold_desc-raw_labels_{atlas}.npy')
    targets = [*map(id2atl.get,targets)]
    mod = [*map(lab2mod.get,targets)]
    # Transition matrix
    mtx = np.zeros((2, 2))
    for ii,current in enumerate(mod):
        if ii==0:
            continue
        previous = mod[ii-1]
        if previous == 'Multimodal' and current == 'Multimodal':
            mtx[0,0] += 1
        elif previous == 'Multimodal' and current == 'Unimodal':
            mtx[0,1] += 1
        elif previous == 'Unimodal' and current == 'Multimodal':
            mtx[1,0] += 1
        elif previous == 'Unimodal' and current == 'Unimodal':
            mtx[1,1] += 1
        else:
            continue
    all_mtx = np.append(all_mtx,mtx.flatten())
mod_df['NTr'] = np.log10(all_mtx)

# %%
lm = sns.lmplot(data=mod_df,x='NTr',y='value',scatter=False,truncate=False,line_kws={'color':'Black','alpha':0.3})
lm.fig.set_dpi(250)
ax = sns.scatterplot(data=mod_df,x='NTr',y='value',hue='transition_type',hue_order=order,palette=pal_list)
sns.despine()
plt.ylabel("Whole-brain ACE [a.u.]")
plt.xlabel("Log$_{10}$(#Transitions)")
r,dof,p,ci,power = pg.rm_corr(data=mod_df,y='NTr',x='value',subject='subject').values[0]
# p = 'p < 0.001' if p<0.001 else f'p={p:.3f}'
# r,p = sp.stats.spearmanr(mod_df['value'],mod_df['NTr'])
ax.legend(title='Transition type',bbox_to_anchor=(1.05,0.4),frameon=False)
ax.set_ylim(ax.get_ylim()[0],800)
lm.ax.set_xlim(1.5,3.5)
plt.title(f"$r_{{rm}}$ = {r:.3f}\nCI: [{ci[0]}, {ci[1]}]",loc='right')
plt.tight_layout()
# plt.savefig('./graphs/modality_transition-no_vs_cost.pdf');

# %%
# plt.figure(dpi=250)
lm = sns.lmplot(data=mod_df,x='NTr',y='value',scatter_kws=dict(s=0),palette='Black')
lm.fig.set_dpi(250)
ax = sns.scatterplot(data=mod_df,x='NTr',y='value',hue='transition_type',hue_order=order)
sns.despine()
plt.ylabel("Brain-wide $E_{control}$")
plt.xlabel("Log$_{10}$(#Transitions)")
r,dof,p,ci,power = pg.rm_corr(data=mod_df,y='NTr',x='value',subject='subject').values[0]
# p = 'p < 0.001' if p<0.001 else f'p={p:.3f}'
# r,p = sp.stats.spearmanr(mod_df['value'],mod_df['NTr'])
ax.legend(title='Transition type',bbox_to_anchor=(1.05,0.4),frameon=False)
plt.title(f"$r_{{rm}}$ = {r:.3f}\nCI: [{ci[0]}, {ci[1]}]",loc='right')
plt.tight_layout()
# plt.savefig('./graphs/modality_transition-no_vs_cost.pdf');





