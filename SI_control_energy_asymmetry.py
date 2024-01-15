# %%
import warnings
warnings.filterwarnings("ignore")
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import pingouin as pg
import matplotlib.pyplot as plt
from neuromaps.images import dlabel_to_gifti
# %%
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

# %% average control energy between states
control_df = pd.read_csv(output_dir + f'optimal-transition-energies_subject-level_{atlas}.csv')

energy_matrix = np.zeros((nlabels-1, nlabels-1))
xticks = []
new_labels = ['VIS', 'SMN', 'DAN', 'SAL', 'LIM', 'FPN', 'DMN']

for i,state1 in enumerate(labels[:-1]):
    xticks.append(new_labels[i])
    for j,state2 in enumerate(labels[:-1]):
        # average E_control for a given intial and goal state
        energy_matrix[i,j] = control_df[(control_df['initial'] == state1) & 
                                        (control_df['goal'] == state2)]['E_control'].mean()

energy_matrix = energy_matrix.T
yticks = xticks        
plt.figure(dpi=150)
sns.heatmap(energy_matrix, cmap='inferno', annot=True, fmt='.2f', 
            xticklabels=xticks, yticklabels=yticks, cbar_kws={'label': 'Control energy [a.u.]'})
        
# %% compute asymmetry in energy matrix
# compute asymmetry in energy matrix
energy_matrix_asym = energy_matrix - energy_matrix.T
np.fill_diagonal(energy_matrix_asym, np.nan)

plt.figure(dpi=150)
sns.heatmap(energy_matrix_asym, cmap='coolwarm', annot=True, fmt='.2f', 
            xticklabels=xticks, yticklabels=yticks, cbar_kws={'label': '$\Delta$ Control energy [a.u.]'})

# %% Compare energy to go from unimodal to heteromodal and vice versa
unimodal_idx = np.where([lab2mod[lab] == 'Unimodal' for lab in labels[:-1]])[0]
heteromodal_idx = np.where([lab2mod[lab] == 'Heteromodal' for lab in labels[:-1]])[0]

np.fill_diagonal(energy_matrix, np.nan)
uni_to_hetero = energy_matrix[unimodal_idx,:][:,heteromodal_idx]
hetero_to_uni = energy_matrix[heteromodal_idx,:][:,unimodal_idx]

plt.figure(dpi=150)
stat = pg.mwu(uni_to_hetero.flatten(), hetero_to_uni.flatten())
u, p = stat['U-val'].values[0], stat['p-val'].values[0]
sns.violinplot(data=[uni_to_hetero.flatten(), hetero_to_uni.flatten()])
plt.xticks([0,1],['Unimodal to Heteromodal', 'Heteromodal to Unimodal'])
plt.ylabel('Control energy [a.u.]')
plt.title(f'Mann-Whitney U = {u:.0f}, p = {p:.2f}')

