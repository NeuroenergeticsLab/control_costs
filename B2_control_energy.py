# %%
import warnings,glob,time,itertools,os
warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
from src.utils import loadmat, compute_optimal_energy_roiwise
from joblib import Parallel,delayed
from nctpy.utils import matrix_normalization

# %% General variables
dataset = 'hcp'
atlas = 'Schaefer'
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
    labels = ['AUD', 'CoP', 'CoPar' ,'DMN', 'DAN', 'FrP', 'None', 'RT', 'SAL','SMH','SMM','VAN','VIS','NOTA']
    atlas_order = pd.read_pickle(root_dir + 'Gordon2016_333_LUT.pkl')
    # A = loadmat('./data/Gordon2016_whole-brain_SC.mat')['connectivity']
    A = np.load(root_dir + 'Gordon2016_333_whole-brain_streamline-density.npy')
    lab2mod={'AUD':'Unimodal','SMH':'Unimodal','SMM':'Unimodal','VIS':'Unimodal',
             'CoP':'Heteromodal','CoPar':'Heteromodal','DMN':'Heteromodal','FrP':'Heteromodal',
             'DAN':'Heteromodal','RT':'Heteromodal','SAL':'Heteromodal','VAN':'Heteromodal',
             'None':'None','NOTA':'Subthreshold'}

elif atlas == 'Glasser':
    labels = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default', 'NOTA']
    atlas_order = pd.read_pickle(root_dir + 'Glasser2016_360_LUT.pkl')
    A = np.load(root_dir + 'Glasser2016_360_whole-brain_streamline-density.npy')
    lab2mod = {'Vis':'Unimodal', 'SomMot':'Unimodal', 'DorsAttn':'Heteromodal', 'SalVentAttn':'Heteromodal', 
               'Limbic':'Heteromodal', 'Cont':'Heteromodal', 'Default':'Heteromodal', 'NOTA':'Subthreshold'} 

elif atlas == 'Schaefer':
    labels = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default', 'NOTA']
    atlas_order = pd.read_pickle(root_dir + 'Schaefer2018_400_LUT.pkl')
    A = np.load(root_dir + 'Schaefer2018_400_whole-brain_streamline-density.npy')
    lab2mod = {'Vis':'Unimodal', 'SomMot':'Unimodal', 'DorsAttn':'Heteromodal', 'SalVentAttn':'Heteromodal', 
               'Limbic':'Heteromodal', 'Cont':'Heteromodal', 'Default':'Heteromodal', 'NOTA':'Subthreshold'}
    
elif atlas == 'Schaefer200':
    labels = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default', 'NOTA']
    atlas_order = pd.read_pickle(root_dir + 'Schaefer2018_200_LUT.pkl')
    lab2mod = {'Vis':'Unimodal', 'SomMot':'Unimodal', 'DorsAttn':'Heteromodal', 'SalVentAttn':'Heteromodal', 
               'Limbic':'Heteromodal', 'Cont':'Heteromodal', 'Default':'Heteromodal', 'NOTA':'Subthreshold'}
    
nrois = len(atlas_order)
nlabels = len(labels)
id2net = dict(zip(np.arange(nlabels)+1,labels))
B = np.eye(nrois)
S = np.eye(nrois)
rho = 1
T = 3

if dataset == 'monash':
    A_norm = matrix_normalization(A,c=1,system='continuous')
else:
    allA = [matrix_normalization(np.load(fn), c=1, system='continuous') \
            for fn in sorted(glob.glob(dataset_dir + f'*streamline*{atlas}.npy'))]

# %% ROI-wise control energy per state
beta_maps = pd.read_csv(output_dir + f'beta-maps_subject-level_{atlas}_{nrois}.csv')
df = pd.DataFrame(columns=['subject','initial','goal','E_control','error'])

for i,subj_id in enumerate(subjects):
    print(f'Starting with subject {subj_id}')
    t = time.time()
    roi_activations = beta_maps[beta_maps['subject']==subj_id]
    if dataset != 'monash':
        A_norm = allA[i]
    tmp_df=pd.DataFrame([],columns=['ROI','network_id'])
    for state1, state2 in itertools.product(labels,labels):
        roi_tmp = atlas_order[['ROI','hem','network_id']].copy()
        roi_tmp['subject'] = [subj_id] * nrois
        roi_tmp['initial'] = [state1] * nrois
        roi_tmp['goal'] = [state2] * nrois
        tmp_df = tmp_df.append(roi_tmp)
    energies, errs = zip(*Parallel(n_jobs=-1,verbose=1)(delayed(compute_optimal_energy_roiwise)(roi_activations,state1,state2,A_norm,T,B,rho,S) for state1, state2 in itertools.product(labels,labels)))
    energies = np.hstack(energies)
    errs = np.hstack(errs)
    tmp_df['E_control'] = energies
    tmp_df['error'] = errs
    df = df.append(tmp_df)
    print(f'Took {time.time()-t:.3f}s')
    df.to_csv(output_dir + f'optimal-transition-energies_roiwise_subject-level_Bmap_{atlas}_{nrois}.csv', index=False)
df.to_csv(output_dir + f'optimal-transition-energies_roiwise_subject-level_Bmap_{atlas}_{nrois}.csv', index=False)

# %% Average control energy

df = pd.read_csv(output_dir + f'optimal-transition-energies_roiwise_subject-level_Bmap_{atlas}_{nrois}.csv')
control_df = pd.DataFrame(columns=['ROI','hem','network_id','E_control','subject'])

for subj_id in subjects:
    print(f'Starting with subject {subj_id}')
    targets = np.load(dataset_dir + f'{subj_id}_task-rest_bold_desc-raw_labels_{atlas}.npy')
    targets = [*map(id2net.get,targets)]
    subj_df = df[df.subject==subj_id]
    roi_energy = np.zeros(nrois)
    tpoints = 0
    for ii,current in enumerate(targets):
        previous = targets[ii-1]
        if ii==0 or 'NOTA' in current:
            continue
        tpoints += 1
        roi_energy += subj_df[(subj_df.initial==previous) & (subj_df.goal==current)]['E_control'].values
    tmp_df = atlas_order[['ROI','hem','network_id']].copy()
    tmp_df['subject'] = [subj_id]*nrois
    tmp_df['E_control'] = roi_energy/tpoints
    control_df = control_df.append(tmp_df)
    control_df.to_csv(output_dir + f'average-control-energy_roiwise_subject-level_Bmap_{atlas}_{nrois}.csv', index=False)
control_df.to_csv(output_dir + f'average-control-energy_roiwise_subject-level_Bmap_{atlas}_{nrois}.csv', index=False)

# %% Compute ACE by modality

df = pd.read_csv(output_dir + f'optimal-transition-energies_roiwise_subject-level_Bmap_{atlas}_{nrois}.csv')
control_df = pd.DataFrame(columns=['subject','ROI','network_id','E_uu','E_hu','E_uh','E_hh','E_sub'])

for subj_id in subjects:
    print(f'Starting with subject {subj_id}')
    targets = np.load(dataset_dir + f'{subj_id}_task-rest_bold_desc-raw_labels_{atlas}.npy')
    targets = [*map(id2net.get,targets)]
    mods = [*map(lab2mod.get,targets)]
    subj_df = df[df.subject==subj_id]
    uu = np.zeros(nrois)
    uh = np.zeros(nrois)
    hu = np.zeros(nrois)
    hh = np.zeros(nrois)
    sub = np.zeros(nrois)
    tuu = 0
    thu = 0
    tuh = 0
    thh = 0
    tsub = 0
    for ii,current in enumerate(targets):
        if ii==0 or current is 'NOTA':
            continue
        previous = targets[ii-1]
        roi_energy = subj_df[(subj_df.initial==previous) & (subj_df.goal==current)]['E_control'].values
        previous_mod = lab2mod[previous]
        current_mod = lab2mod[current]
        if previous_mod == 'Unimodal' and current_mod == 'Unimodal':
            uu += roi_energy
            tuu += 1 
        elif previous_mod == 'Heteromodal' and current_mod == 'Unimodal':
            hu += roi_energy
            thu += 1
        elif previous_mod == 'Unimodal' and current_mod == 'Heteromodal':
            uh += roi_energy
            tuh += 1
        elif previous_mod == 'Heteromodal' and current_mod == 'Heteromodal':
            hh += roi_energy
            thh += 1
        elif current_mod == 'Subthreshold':
            sub += roi_energy
            tsub += 1
        else:
            continue

    tmp_df = atlas_order[['ROI','hem','network_id']].copy()
    tmp_df['subject'] = [subj_id]*nrois
    tmp_df['E_uu'] = uu/tuu
    tmp_df['E_uh'] = uh/tuh
    tmp_df['E_hu'] = hu/thu
    tmp_df['E_hh'] = hh/thh
    tmp_df['E_sub'] = sub/tsub
    
    control_df = control_df.append(tmp_df)
    control_df.to_csv(output_dir + f'average-control-energy_modality_roiwise_subject-level_{atlas}_{nrois}.csv', index=False)
control_df.to_csv(output_dir + f'average-control-energy_modality_roiwise_subject-level_{atlas}_{nrois}.csv', index=False)

