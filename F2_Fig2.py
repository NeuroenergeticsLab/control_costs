# %%
import warnings
warnings.filterwarnings("ignore")
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nilearn.input_data import NiftiLabelsMasker
from nilearn.plotting import plot_surf
from neuromaps.datasets import fetch_fslr
from neuromaps.images import dlabel_to_gifti
from neuromaps.parcellate import Parcellater
from matplotlib.colors import ListedColormap
from src.cmaps import custom_coolwarm, sequential_green
from src.plot_utils import custom_surf_plot

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
id2net = dict(zip(np.arange(nlabels) + 1, labels))

mni = f'./data/MNI152_T1_{den}_brain.nii.gz'

net2id = dict(zip(labels, np.arange(nlabels) + 1))
roi2net = dict(zip(atlas_order['ROI'], atlas_order['network']))

# maskers to parcellate maps
mask = NiftiLabelsMasker(atlas_vol).fit(mni)
surf_masker = Parcellater(atlas_fslr, 'fslr')

# %% Figure 2a
for subj_id in subjects:
    dom_net = np.load(dataset_dir + f'{subj_id}_task-rest_bold_desc-time_labels_{atlas}.npy')
    string_sequence = list(map(id2net.get, dom_net))
    desired_sequence = ['Default', 'DorsAttn', 'SomMot']
    if ''.join(desired_sequence) in ''.join(string_sequence):
        seq_idx = ','.join(string_sequence).find(','.join(desired_sequence))
        start_idx = len(','.join(string_sequence)[:seq_idx].split(',')) - 4
        subject = subj_id
        break
        
ts = np.load(dataset_dir + f'{subject}_task-rest_bold_desc-roi_timeseries_{atlas}.npy')
ts_vec = ts.T.flatten()
nrsn = len(labels) - 1
ntpoints = ts.shape[0]
nrois = ts.shape[1]
roi_order = np.repeat(np.arange(nrois), ntpoints)+1
time_order = np.tile(np.arange(ntpoints), nrois)

ts_df = pd.DataFrame(
    {
        'value': pd.Series(ts_vec, dtype=np.dtype('float')),
        'roi'  : pd.Series(roi_order, dtype=np.dtype('int32')),
        'tr'   : pd.Series(time_order, dtype=np.dtype('int32')),
    })
ts_df['network'] = ts_df['roi'].map(roi2net)

net_avg = np.zeros((ntpoints, nrsn))
for i,net in enumerate(labels[:-1]):
    net_ts = ts_df[ts_df['network']==net].groupby('tr').mean()['value'].values
    net_avg[:,i] = net_ts

colors = custom_coolwarm()
plt.figure(figsize=(8,15), dpi=250)
sns.heatmap(net_avg[start_idx:start_idx+9, :], cmap=colors, cbar_kws={'label':'BOLD [a.u]', 
                                                                      'orientation': 'horizontal',
                                                                      'ticks': []});
plt.axis('off')

# %% Figure 2b
subject = subjects[0]
ts = np.load(dataset_dir + f'{subject}_task-rest_bold_desc-roi_timeseries_{atlas}.npy')
state_labels = np.load(dataset_dir + f'{subject}_task-rest_bold_desc-time_labels_{atlas}.npy')

# find indices of somatomotor state (label 2)
idx = np.where(state_labels == 2)[0]

ts0 = ts[idx[0],:]
ts1 = ts[idx[10],:]
ts2 = ts[idx[20],:]

custom_surf_plot(ts0, parcellation=atlas_fslr, cmap=custom_coolwarm(), cbar_label='BOLD [a.u.]', dpi=250, hemi='left')
custom_surf_plot(ts1, parcellation=atlas_fslr, cmap=custom_coolwarm(), cbar_label='BOLD [a.u.]', dpi=250, hemi='left')
custom_surf_plot(ts2, parcellation=atlas_fslr, cmap=custom_coolwarm(), cbar_label='BOLD [a.u.]', dpi=250, hemi='left')

# %% Figure 2c
states_df = pd.read_csv(f'./results/{dataset}/state-maps_subject-level_{atlas}.csv', index_col=0)
states_df =  states_df.groupby(['ROI', 'state']).mean().reset_index()
avg_states = []
for label in labels[:-1]:
    avg_states.append(states_df[states_df['state'] == label]['value'].values)
avg_states = np.array(avg_states)
surfaces = fetch_fslr(density='32k')
lh, rh = surfaces['inflated']

colors = sns.color_palette('hls', n_colors=nlabels-1)
cw_cmap = custom_coolwarm(N=1000)
bg_int = 0.05

for i,avg_state in enumerate(avg_states):
    net_rois = (atlas_order['network'] == labels[i]).values.astype(int)
    net_rois_surf = surf_masker.inverse_transform(net_rois)
    l_roi, r_roi = net_rois_surf[0].agg_data().astype(int), net_rois_surf[1].agg_data().astype(int)
    
    avg_state_surf = surf_masker.inverse_transform(avg_state)
    l_state, r_state = avg_state_surf[0].agg_data(), avg_state_surf[1].agg_data()
    
    fig, ax = plt.subplots(nrows=1,ncols=2,subplot_kw={'projection': '3d'}, figsize=(8, 4), dpi=200)
    cmap = ListedColormap(['gray', colors[i]])
    plot_surf(lh, l_roi, threshold=-1e-14, cmap=cmap, alpha=1, view='lateral', bg_map=np.ones_like(l_state)*bg_int,
              axes=ax.flat[0], vmin=0.99, vmax=1)
    p = plot_surf(lh, l_state, threshold=-1e-14, cmap=cw_cmap, alpha=1, view='lateral', bg_map=np.ones_like(l_state)*bg_int,
                  colorbar=True, cbar_tick_format='%.0f', axes=ax.flat[1], vmin=-1.5, vmax=1.5)
    p.axes[-1].set_ylabel("BOLD [a.u.]", fontsize=10, labelpad=0.5)
    p.axes[-1].tick_params(labelsize=9, width=0, pad=0.1, labelrotation=90)
    plt.subplots_adjust(wspace=-0.3)
    p.axes[-1].set_position(p.axes[-1].get_position().translated(0.08, 0))
    fig.suptitle(labels[i])

# %% Figure 2d
control_df = pd.read_csv(output_dir + f'optimal-transition-energies_subject-level_{atlas}.csv')
# Plot transitions between these networks
initials = ['Limbic', 'SomMot', 'Vis', 'Vis', 'Default']
goals = ['SomMot', 'Vis', 'Vis', 'Default', 'SalVentAttn']

colors = sequential_green(N=1000)
max_all = 0

for net1, net2 in zip(initials, goals):
    avg_trans = control_df[(control_df['initial']==net1) & (control_df['goal']==net2)].groupby('ROI').mean()['E_control'].values
    max_current = avg_trans.max() # check for maximum value for colorbar
    max_all = max_current if max_current >= max_all else max_all

for net1, net2 in zip(initials, goals):
    avg_trans = control_df[(control_df['initial']==net1) & (control_df['goal']==net2)].groupby('ROI').mean()['E_control'].values   
    avg_trans_surf = surf_masker.inverse_transform(avg_trans)
    l_trans, r_trans = avg_trans_surf[0].agg_data(), avg_trans_surf[1].agg_data()
    l_min, l_max = np.nanmin(l_trans), np.nanmax(l_trans)
    l_trans = np.nan_to_num(l_trans, nan=0)

    fig, ax = plt.subplots(nrows=1,ncols=1,subplot_kw={'projection': '3d'}, figsize=(8, 4), dpi=250)
    plot_surf(lh, l_trans, threshold=-1e-14, cmap=colors, alpha=1, view='lateral',
              colorbar=False, axes=ax, vmin=0, vmax=max_all)
    fig.suptitle(f'{net1} to {net2}')
    
# %% Figure 2e
control_df = pd.read_csv(f'./results/{dataset}/average-control-energy_subject-level_{atlas}.csv',index_col=1)
avg_ctrl = control_df.groupby('ROI').mean()['E_control'].values

custom_surf_plot(avg_ctrl, parcellation=atlas_fslr, cmap=sequential_green(), 
                 cbar_label='TCE [a.u.]', dpi=250, hemi='left')