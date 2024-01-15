# %%
import warnings
warnings.filterwarnings("ignore")
import os
import glob
import numpy as np
import pandas as pd
from src.utils import tr_net_labels
from joblib import Parallel,delayed

# %% General variables

dataset = 'hcp'
atlas = 'Schaefer400'
root_dir = './data/'
dataset_dir = root_dir + f'{dataset}/'
output_dir = f'./results/{dataset}/'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

subjects = sorted([fn.split('/')[-1][:10] 
                   for fn in glob.glob(dataset_dir + f'sub*timeseries_{atlas}*.npy')])

if atlas == 'Gordon':
    labels = ['AUD', 'CoP', 'CoPar' ,'DMN', 'DAN', 'FrP', 'None', 'RT', 
              'SAL', 'SMH', 'SMM', 'VAN', 'VIS', 'NOTA']
    atlas_order = pd.read_pickle(root_dir + 'Gordon2016_333_LUT.pkl')
    
elif atlas == 'Schaefer400':
    labels = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 
              'Default', 'NOTA']
    atlas_order = pd.read_pickle(root_dir + 'Schaefer2018_400_LUT.pkl')

elif atlas == 'Schaefer200':
    labels = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 
              'Default', 'NOTA']
    atlas_order = pd.read_pickle(root_dir + 'Schaefer2018_200_LUT.pkl')

nlabels = len(labels)
nrois = len(atlas_order)
activity_thr = 0.5

net2id = dict(zip(labels, np.arange(nlabels)+1))
roi2net = dict(zip(atlas_order['ROI'], atlas_order['network']))

# %% Create state maps
group_states = pd.DataFrame(columns=['subject','state','value'])

for subj_id in subjects:
    print(f'Starting with subject {subj_id}')
    # load time series and format into tall dataframe format
    ttss = np.load(dataset_dir + f'{subj_id}_task-rest_bold_desc-roi_timeseries_{atlas}.npy')
    
    ttss_vec = ttss.T.flatten()
    nrois = ttss.shape[1]

    roi_order = np.repeat(np.arange(ttss.shape[1]),ttss.shape[0])+1
    time_order = np.tile(np.arange(ttss.shape[0]),ttss.shape[1])

    # each row corresponts to a regional value at a given TR
    ttss_df = pd.DataFrame(
        {
            'value': pd.Series(ttss_vec, dtype=np.dtype('float')),
            'roi'  : pd.Series(roi_order, dtype=np.dtype('int32')),
            'tr'   : pd.Series(time_order, dtype=np.dtype('int32')),
        })
    ttss_df['network'] = ttss_df['roi'].map(roi2net)
    ttss_df['network_id'] = ttss_df['network'].map(net2id)

    # for each TR calculate the network with the most (positively) active ROIs
    time_labels = Parallel(n_jobs=-1)       \
                    (delayed(tr_net_labels) \
                    (ttss_df[ttss_df.tr==tr], nlabels, activity_thr) 
                    for tr in range(len(ttss)))
    time_labels = np.array(time_labels)

    np.save(dataset_dir + f'{subj_id}_task-rest_bold_desc-time_labels_{atlas}.npy', time_labels)
    
    # create state-wise time mask where ones indicate the time points of dominance
    state_masks = pd.DataFrame({lab:(time_labels==lab_idx+1) 
                                for lab_idx,lab in enumerate(labels)}).values

    # calculate the mean activity across dominant time points for each state
    for i,mask in enumerate(state_masks.T):
        print(f'State: {labels[i]}')
        state = np.mean(ttss[mask],axis=0)

        tmp_df = atlas_order[['ROI','network_id']].copy()
        tmp_df['subject'] = [f'{subj_id}'] * nrois
        tmp_df['state'] = [labels[i]] * nrois
        tmp_df['value'] = state
        group_states = group_states.append(tmp_df)
        group_states.to_csv(output_dir + f'state-maps_subject-level_{atlas}.csv', index=False)

group_states.to_csv(output_dir + f'state-maps_subject-level_{atlas}.csv', index=False)