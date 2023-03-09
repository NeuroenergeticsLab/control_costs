#####################################################################################################################################
# %% Script to show statistics and plots using the Monash dataset
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

dataset = 'monash'
root_dir = './data/'
dataset_dir = root_dir + f'{dataset}/'
output_dir = f'./results/{dataset}/'

if dataset == 'hcp':
    subjects = sorted([fn.split('/')[-1][:10] for fn in glob.glob(dataset_dir + 'sub*.npy')])
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
    lab2mod = {'Vis':'Unimodal', 'SomMot':'Unimodal', 'DorsAttn':'Multimodal', 'SalVentAttn':'Multimodal', 
               'Limbic':'Multimodal', 'Cont':'Multimodal', 'Default':'Multimodal', 'NOTA':'Subthreshold'}

nrois = len(atlas_order)
id2atl=dict(zip(np.arange(nlabels)+1,labels))
mni = f'./data/MNI152_T1_{den}_brain.nii.gz'
mask = NiftiLabelsMasker(atlas_vol).fit(mni)

# CMRglc map
tum_cmrglc = './data/avg_cmrglc_subjs-20_45min_mcf_fwhm-6_quant-cmrglc_acq-2242min_pvc-pveseg_mni-3mm.nii.gz'
roi_cmrglc = mask.fit_transform(tum_cmrglc)

# CMRO2 map
cmrox_164k = fetch_annotation(source='raichle', desc='cmr02', space='fsLR', res='164k', data_dir=root_dir)
#### cmrox = fslr_to_fslr(cmrox_164k,'32k')
surf_masker = Parcellater(atlas_fslr,'fslr')
#### roi_cmrox = surf_masker.fit_transform(cmrox,'fslr')

# %%
cmrox = ('/home/tumnic/eceballos/neuromaps-data/annotations/raichle/cmr02/fsLR/source-raichle_desc-cmr02_space-fsLR_den-32k_hemi-L_feature.func.gii.gz',
         '/home/tumnic/eceballos/neuromaps-data/annotations/raichle/cmr02/fsLR/source-raichle_desc-cmr02_space-fsLR_den-32k_hemi-R_feature.func.gii.gz')
roi_cmrox = surf_masker.fit_transform(cmrox,'fslr')

# %% Compare spatial maps
##### control_df = pd.read_csv(f'./results/{dataset}/average-optimal-control-energy_roiwise_subject-level_Bmap_{atlas}_{nrois}.csv',index_col=1)
control_df = pd.read_csv('./results/monash_nomapper/average-optimal-control-energy_roiwise_subject-level_Bmap_Gordon_333.csv',index_col=1)
##### avg_ctrl = control_df.groupby('ROI').mean()['E_control'].values
avg_ctrl = control_df.groupby('ParcelID').mean()['E_control'].values
nulls = burt2020(avg_ctrl,atlas='MNI152',density=den,n_perm=1000,parcellation=atlas_vol,seed=0)

# %%
glc_corr, p_glc = compare_images(avg_ctrl,roi_cmrglc,nulls=nulls)
ox_corr, p_ox = compare_images(avg_ctrl,roi_cmrox,nulls=nulls)

glc_corr_nulls = []
ox_corr_nulls = []
for ar in nulls.T:
    glc_corr_nulls.append(pearsonr(ar,roi_cmrglc[0])[0])
    ox_corr_nulls.append(pearsonr(ar,roi_cmrox)[0])
    
with sns.color_palette("tab10"):
    plt.figure(dpi=100)
    ax = sns.histplot(glc_corr_nulls,stat='density',kde=True,alpha=0.4)
    ax.axvline(glc_corr,color=sns.color_palette("Set2")[1])
    ax.set_xlabel("Pearson's $r$",fontsize='large')
    bottom,top = plt.ylim()
    plt.text(glc_corr+0.02,bottom+1,f'$r$  =  {glc_corr:.3f}\n$p_{{spin}}$ = {p_glc:.3f}',fontsize='small')
    plt.legend(['Null data','Observation'],bbox_to_anchor=(0.8,1),frameon=False)
    sns.despine()
    plt.title('Monash and $CMR_{glc}$')
    
    plt.figure(dpi=100)
    ax = sns.histplot(ox_corr_nulls,stat='density',kde=True,alpha=0.4)
    ax.axvline(ox_corr,color=sns.color_palette("Set2")[3])
    ax.set_xlabel("Pearson's $r$",fontsize='large')
    bottom,top = plt.ylim()
    plt.text(ox_corr+0.02,bottom+1,f'$r$  =  {ox_corr:.3f}\n$p_{{spin}}$ = {p_ox:.3f}',fontsize='small')
    plt.legend(['Null data','Observation'],bbox_to_anchor=(0.8,1),frameon=False)
    sns.despine()
    plt.title('Monash and $CMR_{O2}$')
#     plt.savefig('./graphs/avg-E_control_spin.pdf')

# %% [markdown]
# #### Compare individual glucose uptake and average control energy

# %%
control_df = pd.read_csv('./results/monash_nomapper/average-optimal-control-energy_roiwise_subject-level_Bmap_Gordon_333.csv')
# sub_ctrl = control_df['E_control'].values

sub_pet = []
for subj_id in subjects:
    pet_brain = dataset_dir + f'{subj_id}/pet/{subj_id}_task-rest_pet_mcf_brain_smooth_8_mean_suv_mni-2mm.nii.gz'
    sub_pet.append(mask.fit_transform(pet_brain))
control_df['PET'] = np.ravel(sub_pet)

# %%
plot_df = control_df.groupby('ParcelID').mean()
sns.lmplot(data=plot_df,x='E_control',y='PET',)

# %%
control_df['Mod'] = control_df['Community'].map(lab2mod)
g = sns.lmplot(data=control_df[~control_df['Mod'].isin(['None'])],x='E_control',y='PET',col='subject',sharey=False)

def annotate(data, **kws):
    r, p = pearsonr(data['E_control'], data['PET'])
    ax = plt.gca()
    ax.text(.05, .8, 'r={:.2f}, p={:.2g}'.format(r, p),
            transform=ax.transAxes)
    
g.map_dataframe(annotate)

# %%
labels_subset = ['AUD', 'CoP', 'CoPar', 'DMN', 'DAN', 'FrP', 'RT', 'SAL', 'SMH', 'SMM', 'VAN', 'VIS']
all_r = []

for lab in labels_subset:
    r = []
    for subj_id in subjects:
        tmp = control_df[(control_df['Community']==lab) & (control_df['subject']==subj_id)]
        r.append(pearsonr(tmp['E_control'],tmp['PET'])[0])
    all_r.append(r)
all_r = np.array(all_r).T
plot_df = pd.DataFrame(all_r, columns=labels_subset)
plot_df = pd.melt(plot_df)

# %%
from scipy.stats import ttest_1samp
ps = []
for lab in ['Unimodal','Multimodal']:
    ps.append(ttest_1samp(plot_df[plot_df['variable']==lab]['value'],0)[1])

significant, p_corr = pg.multicomp(ps,alpha=0.01)
print(np.array(labels_subset)[significant])
print(p_corr[significant])

# %%
with sns.axes_style(style="whitegrid"):
    plt.figure(figsize=(9,6),dpi=150)
    ax = sns.boxplot(data=plot_df,x='variable',y='value',width=0.3,linewidth=1.5)
    sns.stripplot(data=plot_df,x='variable',y='value',alpha=0.3,ax=ax,linewidth=1)
    for x_pos in np.nonzero(significant)[0]:
        plt.text(x_pos, plot_df[plot_df['variable']==labels_subset[x_pos]]['value'].max()+0.1, "***", ha='center', va='bottom', color='k')
    sns.despine(offset=15,bottom=True,left=True)
    ax.set_xlabel('Network')
    ax.set_ylabel("Pearson's r");

# %%
control_df['Mod'] = control_df['Community'].map(lab2mod)
mods = ['Unimodal','Multimodal']
all_r = []

for lab in mods:
    r = []
    for subj_id in subjects:
        tmp = control_df[(control_df['Mod']==lab) & (control_df['subject']==subj_id)]
        r.append(pearsonr(tmp['E_control'],tmp['PET'])[0])
    all_r.append(r)
all_r = np.array(all_r).T
plot_df = pd.DataFrame(all_r, columns=['Unimodal','Multimodal'])
plot_df = pd.melt(plot_df)

# %%
from scipy.stats import ttest_1samp
ps = []
for lab in mods:
    ps.append(ttest_1samp(plot_df[plot_df['variable']==lab]['value'],0)[1])

significant, p_corr = pg.multicomp(ps)
# significant, p_corr = pg.multicomp(ps,alpha=0.01)
# print(np.array(labels_subset)[significant])
# print(p_corr[significant])

# %%
with sns.axes_style(style="whitegrid"):
    plt.figure(figsize=(9,6),dpi=150)
    ax = sns.boxplot(data=plot_df,x='variable',y='value',width=0.3,linewidth=1.5)
    sns.stripplot(data=plot_df,x='variable',y='value',alpha=0.3,ax=ax,linewidth=1)
    for x_pos in np.nonzero(significant)[0]:
        plt.text(x_pos, plot_df[plot_df['variable']==mods[x_pos]]['value'].max()+0.1, "***", ha='center', va='bottom', color='k')
    sns.despine(offset=15,bottom=True,left=True)
    ax.set_xlabel('Network')
    ax.set_ylabel("Pearson's r");

# %% [markdown]
# ### Average control energy grouped by modality

# %%
##### mod_df = pd.read_csv(f'./results/{dataset}/average-optimal-control-energy_modality_roiwise_subject-level_Bmap_{atlas}_{nrois}.csv').iloc[:,-6]
mod_df = pd.read_csv('./results/monash_nomapper/average-optimal-control-energy_modality_roiwise_subject-level_Bmap_Gordon_333.csv').iloc[:,-6:]

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
intra = mod_df[(mod_df.transition_type=='M→M')|(mod_df.transition_type=='U→U')]['value'].values
inter = mod_df[(mod_df.transition_type=='U→M')|(mod_df.transition_type=='M→U')]['value'].values
T,pval = ttest_rel(inter, intra)

plt.figure(dpi=250)
with sns.axes_style(style="whitegrid"):
    ax = sns.boxplot(data=mod_df,x='transition_type',y='value',width=0.3,linewidth=1.5,order=order)
    sns.stripplot(data=mod_df,x='transition_type',y='value',alpha=0.1,ax=ax,linewidth=1,order=order)
    sns.despine(ax=ax,bottom=True)
    ax.set_title(f'Monash\n T={T:.1f}, p={pval:.1e}')
    ax.tick_params(tick1On=False)
    plt.xlabel("Transition type")
    plt.ylabel("Brain-wide $E_{control}$")
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    # plt.savefig('./graphs/modality_transition_cost.pdf')

# %% [markdown]
# ### Contrast average control energy to number of transitions

# %%
# Number of transitions
all_mtx = np.array([])

for subj_id in subjects:
    if dataset == 'hcp':
        targets = np.load(dataset_dir + f'{subj_id}_raw_labels.npy')
    elif dataset == 'monash':
        targets = np.load(dataset_dir + f'{subj_id}/func/{subj_id}_raw_labels.npy')
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
plt.savefig('./graphs/modality_transition-no_vs_cost.pdf');

# %%
intra = mod_df[(mod_df.transition_type=='M→M')|(mod_df.transition_type=='U→U')]['value'].values
inter = mod_df[(mod_df.transition_type=='U→M')|(mod_df.transition_type=='M→U')]['value'].values
sp.stats.ttest_rel(inter, intra)
# pg.ttest(intra, inter, paired=True,alternative='less')