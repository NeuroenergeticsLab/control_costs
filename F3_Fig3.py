# %% 
import warnings
warnings.filterwarnings("ignore")
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import pingouin as pg
import matplotlib.pyplot as plt
from nilearn.input_data import NiftiLabelsMasker
from neuromaps.images import dlabel_to_gifti
from neuromaps.parcellate import Parcellater
from matplotlib.gridspec import GridSpec
from src.utils import get_significance_string
from src.cmaps import sequential_green
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
    labels = ['AUD', 'CoP', 'CoPar', 'DMN', 'DAN', 'FrP', 'None', 'RT', 'SAL', 'SMH', 'SMM', 'VAN', 'VIS', 'NOTA']
    lab_short = labels
    atlas_order = pd.read_pickle(root_dir + 'Gordon2016_333_LUT.pkl')
    atlas_fslr = dlabel_to_gifti(root_dir + 'Gordon2016_333_space-fsLR_den-32k.dlabel.nii')
    lab2mod = {'AUD': 'Unimodal', 'SMH': 'Unimodal', 'SMM': 'Unimodal', 'VIS': 'Unimodal',
               'CoP': 'Heteromodal', 'CoPar': 'Heteromodal', 'DMN': 'Heteromodal', 'FrP': 'Heteromodal',
               'DAN': 'Heteromodal', 'RT': 'Heteromodal', 'SAL': 'Heteromodal', 'VAN': 'Heteromodal',
               'None': 'None', 'NOTA': 'Subthreshold'}

elif atlas == 'Schaefer400':
    den = '2mm'
    atlas_vol = root_dir + f'Schaefer2018_400_7N_space-MNI152_den-{den}.nii.gz'
    labels = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default', 'NOTA']
    lab_short = ['VIS', 'SMN', 'DAN', 'SAL', 'LIM', 'FPN', 'DMN', 'NOTA']
    atlas_order = pd.read_pickle(root_dir + 'Schaefer2018_400_LUT.pkl')
    atlas_fslr = dlabel_to_gifti(root_dir + 'Schaefer2018_400_7N_space-fsLR_den-32k.dlabel.nii')
    lab2mod = {'Vis': 'Unimodal', 'SomMot': 'Unimodal', 'DorsAttn': 'Heteromodal', 'SalVentAttn': 'Heteromodal',
               'Limbic': 'Heteromodal', 'Cont': 'Heteromodal', 'Default': 'Heteromodal', 'NOTA': 'Subthreshold'}

elif atlas == 'Schaefer200':
    den = '2mm'
    atlas_vol = root_dir + f'Schaefer2018_200_7N_space-MNI152_den-{den}.nii.gz'
    labels = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default', 'NOTA']
    lab_short = ['VIS', 'SMN', 'DAN', 'SAL', 'LIM', 'FPN', 'DMN', 'NOTA']
    atlas_order = pd.read_pickle(root_dir + 'Schaefer2018_200_LUT.pkl')
    atlas_fslr = dlabel_to_gifti(root_dir + 'Schaefer2018_200_7N_space-fsLR_den-32k.dlabel.nii')
    lab2mod = {'Vis': 'Unimodal', 'SomMot': 'Unimodal', 'DorsAttn': 'Heteromodal', 'SalVentAttn': 'Heteromodal',
               'Limbic': 'Heteromodal', 'Cont': 'Heteromodal', 'Default': 'Heteromodal', 'NOTA': 'Subthreshold'}

nrois = len(atlas_order)
nlabels = len(labels)
id2net = dict(zip(np.arange(nlabels)+1,labels))
roi2netid = dict(zip(atlas_order['ROI'],atlas_order['network_id']))
mni = f'./data/MNI152_T1_{den}_brain.nii.gz'

# maskers to parcellate maps
mask = NiftiLabelsMasker(atlas_vol).fit(mni)
surf_masker = Parcellater(atlas_fslr,'fslr')

# %% Figure 3a
plot = True
control_df = pd.read_csv(output_dir + f'average-control-energy_subject-level_{atlas}.csv')
avg_ctrl = control_df.groupby('ROI').mean()['E_control'].values

if plot:
    custom_surf_plot(avg_ctrl, space='fsLR', density='32k', template='inflated', dpi=250,
                    parcellation=atlas_fslr, cbar_label='TCE [a.u.]', cmap=sequential_green())

# %% Figure 3b
data_info = pd.read_csv(dataset_dir + 'participants.csv')
data_info['subject'] = data_info['subject'].apply(lambda x: f'sub-{str(x)}')
sub2sex = dict(zip(data_info['subject'], data_info['gender']))
control_df['sex'] = control_df['subject'].map(sub2sex)
plot_df = control_df.groupby(['sex','hem','ROI']).mean().reset_index()
plot_df['network_id'] = plot_df['ROI'].map(roi2netid)
plot_df['network'] = plot_df['network_id'].map(id2net)
plot_df['mod'] = plot_df['network'].map(lab2mod)

fig = plt.figure(figsize=(8, 4), dpi=250)
gs = GridSpec(1, 2, width_ratios=[6, 2])
height_percentage = 1.05
lwidth = 0.8
bar_lwidth = 1.5
alpha = 0.3

# Plot networks
ax1 = plt.subplot(gs[0])
mean_values = plot_df.groupby('network_id')['E_control'].mean()
ids = mean_values.sort_values(ascending=False).index.astype(int)
order = [labels[i-1] for i in ids]
order_short = [lab_short[i-1] for i in ids]
colors = sequential_green(return_palette=True)
cmap = [colors[0] , colors[3]]
sns.barplot(data=plot_df, x='network', y='E_control', 
            errwidth=bar_lwidth, ci='sd', ax=ax1, palette=cmap, order=order, hue='hem')
sns.stripplot(data=plot_df, x='network', y='E_control', 
                alpha=0.1, linewidth=lwidth, ax=ax1, palette=cmap, order=order, dodge=True, hue='hem')
ax1.legend_.remove()

# Compare hemispheres
ax2 = plt.subplot(gs[1])
sns.barplot(data=plot_df, x='hem', y='E_control', ci='sd', ax=ax2, palette=cmap, 
            errwidth=bar_lwidth, order=['L', 'R'])

middle_x1 = ax2.get_xlim()[0] + (ax2.get_xlim()[1] - ax2.get_xlim()[0]) / 4
middle_x2 = ax2.get_xlim()[0] + 3 * (ax2.get_xlim()[1] - ax2.get_xlim()[0]) / 4

ax2.plot([middle_x1, middle_x2], 
            [ax2.get_ylim()[1]*height_percentage, ax2.get_ylim()[1]*height_percentage], 
            color='black', linewidth=1.5)

# Mann-Whitney U-Test 
results = pg.mwu(plot_df[plot_df['hem']=='L']['E_control'], plot_df[plot_df['hem']=='R']['E_control'])
p_str = get_significance_string(results['p-val'][0], type='asterisk')
ax2.annotate(p_str, xycoords='data', ha='center', 
                xy=((middle_x1 + middle_x2)/2, ax2.get_ylim()[1]*(height_percentage)),
                xytext=((middle_x1 + middle_x2)/2, ax2.get_ylim()[1]*(height_percentage)))

sns.despine()
ax1.set(xlabel=None, xticklabels=order_short)
ax1.set_ylabel('TCE [a.u.]', fontsize=14)
ax2.spines['left'].set_visible(False)
ax2.set(yticklabels=[], xlabel=None, ylabel=None)
ax2.tick_params(left=False)
ax2.set_ylim(ax1.get_ylim())
plt.subplots_adjust(wspace=0.1)
plt.tight_layout()

# print results
print(f"U = {results['U-val'][0]}")
print(f"P = {results['p-val'][0]}")

# %% Figure 3c
data_info = pd.read_csv(dataset_dir + 'participants.csv')
data_info['subject'] = data_info['subject'].apply(lambda x: f'sub-{str(x)}')
sub2age = dict(zip(data_info['subject'], data_info['age']))
control_df['age'] = control_df['subject'].map(sub2age)
plot_df = control_df.groupby(['age','hem','ROI']).mean().reset_index()
plot_df['network'] = plot_df['network_id'].map(id2net)
plot_df['mod'] = plot_df['network'].map(lab2mod)

fig = plt.figure(figsize=(8, 4), dpi=250)
gs = GridSpec(1, 2, width_ratios=[6, 2])
height_percentage = 1.05
lwidth = 0.8
bar_lwidth = 1.5
alpha = 0.3

# Plot networks
ax1 = plt.subplot(gs[0])
mean_values = plot_df.groupby('network_id')['E_control'].mean()
ids = mean_values.sort_values(ascending=False).index.astype(int)
order = [labels[i-1] for i in ids]
order_short = [lab_short[i-1] for i in ids]
colors = sequential_green(return_palette=True)
cmap = [colors[1], colors[2], colors[4], colors[5]]
sns.barplot(data=plot_df, x='network', y='E_control',
            errwidth=bar_lwidth, ci='sd', ax=ax1, palette=cmap, order=order, hue='age')
sns.stripplot(data=plot_df, x='network', y='E_control',
                alpha=0.1, linewidth=lwidth, ax=ax1, palette=cmap, order=order, dodge=True, hue='age')
ax1.legend_.remove()

# Compare ages
ax2 = plt.subplot(gs[1])
sns.barplot(data=plot_df, x='age', y='E_control', ci='sd', ax=ax2, palette=cmap,
            errwidth=bar_lwidth, order=['22-25', '26-30', '31-35', '36+'])
plt.xticks(rotation=45)

middle_x1 = ax2.get_xlim()[0] + (ax2.get_xlim()[1] - ax2.get_xlim()[0]) / 4
middle_x2 = ax2.get_xlim()[0] + 3 * (ax2.get_xlim()[1] - ax2.get_xlim()[0]) / 4

ax2.plot([middle_x1, middle_x2], 
            [ax2.get_ylim()[1]*height_percentage, ax2.get_ylim()[1]*height_percentage], 
            color='black', linewidth=1.5)

# test group differences with ANOVA
# first check variance homogeneity
levene = pg.homoscedasticity(data=plot_df, dv='E_control', group='age')
if levene['equal_var'][0]:
    results = pg.anova(data=plot_df, dv='E_control', between='age')
else:
    results = pg.welch_anova(data=plot_df, dv='E_control', between='age')

p_str = get_significance_string(results['p-unc'][0], type='asterisk')
ax2.annotate(p_str, xycoords='data', ha='center', 
                xy=((middle_x1 + middle_x2)/2, ax2.get_ylim()[1]*(height_percentage)),
                xytext=((middle_x1 + middle_x2)/2, ax2.get_ylim()[1]*(height_percentage)))

sns.despine()
ax1.set(xlabel=None, xticklabels=order_short)
ax1.set_ylabel('TCE [a.u.]', fontsize=14)
ax2.spines['left'].set_visible(False)
ax2.set(yticklabels=[], xlabel=None, ylabel=None)
ax2.tick_params(left=False)
ax2.set_ylim(ax1.get_ylim())
plt.subplots_adjust(wspace=0.1)

# print levene and anova results
print(f"W = {levene['W'][0]}")
print(f"P = {levene['pval'][0]}")

print(f"F = {results['F'][0]}")
print(f"P = {results['p-unc'][0]}")
