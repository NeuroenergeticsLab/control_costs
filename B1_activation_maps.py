# %%
import warnings
warnings.filterwarnings("ignore")
import os
import glob
import numpy as np
import pandas as pd
from src.utils import tr_net_labels
from joblib import Parallel,delayed
from pingouin import linear_regression

# %% General variables

dataset = 'hcp'
atlas = 'Schaefer200'
root_dir = './data/'
dataset_dir = root_dir + f'{dataset}/'
output_dir = f'./results/{dataset}/'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

if dataset == 'tum':
    subjects = sorted([fn.split('/')[-1][:7] for fn in glob.glob(dataset_dir + f'sub*timeseries_{atlas}*.npy')])
elif dataset == 'hcp':
    subjects = sorted([fn.split('/')[-1][:10] for fn in glob.glob(dataset_dir + f'sub*timeseries_{atlas}*.npy')])
elif dataset == 'mica':
    subjects = sorted([fn.split('/')[-1][:9] for fn in glob.glob(dataset_dir + f'sub*timeseries_{atlas}*.npy')])
elif dataset == 'monash':
    subjects = [f'sub-{sid:02}' for sid in np.arange(1,28)]

if atlas == 'Gordon':
    labels = ['AUD', 'CoP', 'CoPar' ,'DMN', 'DAN', 'FrP', 'None', 'RT', 'SAL','SMH','SMM','VAN','VIS','NOTA']
    atlas_order = pd.read_pickle(root_dir + 'Gordon2016_333_LUT.pkl')
    
elif atlas == 'Glasser':
    labels = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default', 'NOTA']
    atlas_order = pd.read_pickle(root_dir + 'Glasser2016_360_LUT.pkl')
    
elif atlas == 'Schaefer':
    labels = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default', 'NOTA']
    atlas_order = pd.read_pickle(root_dir + 'Schaefer2018_400_LUT.pkl')
    
elif atlas == 'Schaefer100':
    labels = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default', 'NOTA']
    atlas_order = pd.read_pickle(root_dir + 'Schaefer2018_100_LUT.pkl')

elif atlas == 'Schaefer200':
    labels = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default', 'NOTA']
    atlas_order = pd.read_pickle(root_dir + 'Schaefer2018_200_LUT.pkl')

nlabels = len(labels)
nrois = len(atlas_order)
activity_thr = 0.5

net2id = dict(zip(labels, np.arange(nlabels)+1))
roi2net = dict(zip(atlas_order['ROI'], atlas_order['network']))

# %%
group_betas = pd.DataFrame(columns=['subject','state','beta'])

for subj_id in subjects:
    print(f'Starting with subject {subj_id}')
    ttss = np.load(dataset_dir + f'{subj_id}_task-rest_bold_desc-roi_timeseries_{atlas}.npy')
    
    ttss_vec = ttss.T.flatten()
    nrois = ttss.shape[1]

    roi_order = np.repeat(np.arange(ttss.shape[1]),ttss.shape[0])+1
    time_order = np.tile(np.arange(ttss.shape[0]),ttss.shape[1])

    ttss_df = pd.DataFrame(
        {
            'value': pd.Series(ttss_vec, dtype=np.dtype('float')),
            'roi'  : pd.Series(roi_order, dtype=np.dtype('int32')),
            'tr'   : pd.Series(time_order, dtype=np.dtype('int32')),
        })
    ttss_df['network'] = ttss_df['roi'].map(roi2net)
    ttss_df['network_id'] = ttss_df['network'].map(net2id)

    # for each TR calculate the network with the most (positively) active ROIs
    targets = Parallel(n_jobs=-1)(delayed(tr_net_labels)(ttss_df[ttss_df.tr==tr],nlabels,activity_thr) for tr in range(len(ttss)))
    targets = np.array(targets)

    np.save(dataset_dir + f'{subj_id}_task-rest_bold_desc-raw_labels_{atlas}.npy',targets)
    
    # one-hot-encode states
    states = pd.DataFrame({lab:(targets==lab_idx+1).astype(int) for lab_idx,lab in enumerate(labels)}).values

    # calculate beta maps
    for i,state in enumerate(states.T):
        if state.sum()==0:
            continue
        print(f'State: {labels[i]}')
        betas = []
        for roi in range(ttss.shape[1]):
            model = linear_regression(state,ttss[:,roi])
            betas.append(model['coef'][1])
        betas = np.array(betas)

        tmp_df = atlas_order[['ROI','network_id']].copy()
        tmp_df['subject'] = [f'{subj_id}'] * nrois
        tmp_df['state'] = [labels[i]] * nrois
        tmp_df['beta'] = betas
        group_betas = group_betas.append(tmp_df)
        group_betas.to_csv(output_dir + f'beta-maps_subject-level_{atlas}_{nrois}.csv')
        
group_betas.to_csv(output_dir + f'beta-maps_subject-level_{atlas}_{nrois}.csv')
# %%
