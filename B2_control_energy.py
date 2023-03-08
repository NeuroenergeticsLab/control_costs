# %%
import warnings
warnings.filterwarnings("ignore")
import glob,time,itertools
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy as sp
from joblib import Parallel,delayed
from nilearn.input_data import NiftiLabelsMasker
from network_control.energies import minimum_input, integrate_u
from network_control.utils import matrix_normalization
from neuromaps.datasets import fetch_annotation
from neuromaps import stats, nulls, resampling

# %% Functions

def loadmat(filename):
    """
    This function should be called instead of direct scipy.io.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects.
    Taken from https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries
    """
    data = sp.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    """
    Checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries.
    """
    for key in dict:
        if isinstance(dict[key], sp.io.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        

def _todict(matobj):
    """
    A recursive function which constructs from matobjects nested dictionaries.
    """
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, sp.io.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

def compute_optimal_energy_roiwise(df,state1,state2,A_norm,T,B,rho,S):
    """
    Computes optimal control energy to transition from state1 to state2.
    """
    import numpy as np
    import pandas as pd
    from network_control.energies import optimal_input
    from scipy import integrate
    nrois = len(A_norm)
    initial = df[df['state'] == state1]['beta'].values[:,None]
    goal = df[df['state'] == state2]['beta'].values[:,None]
    if np.array_equal(initial,np.empty([0,1])) or np.array_equal(goal,np.empty([0,1])):
        energy, n_err = np.repeat(np.nan,nrois),np.repeat(np.nan,nrois)
    else:
        x, u, n_err = optimal_input(A_norm,T,B,initial,goal,rho,S)
        energy = integrate_u(u)
        n_err = np.repeat(n_err,nrois)
    return energy, n_err

# %% General variables

dataset = 'hcp'
root_dir = './data/'
dataset_dir = root_dir + f'{dataset}/'
output_dir = f'./results/{dataset}/'

if dataset == 'hcp':
    subjects = sorted([fn.split('/')[-1][:10] for fn in glob.glob(dataset_dir + 'sub*timeseries*.npy')])
elif dataset == 'monash':
    subjects = [f'sub-{sid:02}' for sid in np.arange(1,28)]

atlas = 'Gordon'
if atlas == 'Gordon':
    labels = ['AUD', 'CoP', 'CoPar' ,'DMN', 'DAN', 'FrP', 'None', 'RT', 'SAL','SMH','SMM','VAN','VIS','NOTA']
    atlas_fn = root_dir + 'Gordon2016_space-MNI152_den-3mm.nii'
    atlas_order = pd.read_csv(root_dir + 'gordon2016_parcels.csv')
    atlas_order = atlas_order.drop(columns=[atlas_order.columns[-1],'Surface area (mm2)','Centroid (MNI)'])
    nlabels = len(labels)
    atl2id = dict(zip(labels,np.arange(nlabels)+1))
    atl_id2label = dict(zip(atlas_order['ParcelID'].tolist(), atlas_order['Community']))
    atlas_order['CommunityID'] = atlas_order['Community'].map(atl2id)
    A = loadmat(root_dir + 'Gordon2016_whole-brain_SC.mat')['connectivity']
        
elif atlas == 'Schaefer':
    labels = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default', 'NOTA']
    atlas_fn = root_dir + 'Schaefer2018_400_7N_space-MNI152_den-2mm.nii.gz'
    atlas_order = pd.read_csv(root_dir + 'Schaefer2018_400_7N_order.txt',sep='\t',header=None)
    nlabels = len(labels)
    atlas_order['network'] = atlas_order[1].str.split('_').str.get(2)
    atl2id = dict(zip(atlas_order['network'].unique(), range( 1, nlabels+1 )))
    atl_id2label = dict(zip(atlas_order[0].tolist(), atlas_order['network']))
    atlas_order['network_id'] = atlas_order['network'].map(atl2id)
    A = loadmat(root_dir + 'Schaefer2018_400_7N_whole-brain_SC.mat')['connectivity']
    
A_norm = matrix_normalization(A,c=1,version='continuous')
B = np.eye(A.shape[0])
S = np.eye(A.shape[0])
rho = 1
T = 4


states = labels
nstates = len(states)
nrois = len(A)

# mni = './data/MNI152_T1_3mm_brain.nii.gz' if atlas == 'Gordon' else './data/MNI152_T1_2mm_brain.nii.gz'
# bin_mask = './data/MNI152_T1_3mm_GM.nii.gz' if atlas == 'Gordon' else './data/MNI152_T1_2mm_GM.nii.gz'
# mask = NiftiLabelsMasker(atlas_fn,mask_img=bin_mask).fit(mni)
# cmrglc_map = './data/avg_cmrglc_subjs-20_45min_mcf_fwhm-6_quant-cmrglc_acq-2242min_pvc-pveseg_mni-3mm.nii.gz'
# roi_cmrglc = mask.transform(cmrglc_map)[0]

# %% ROI-wise control energy per state

roi_all = pd.read_csv(output_dir + f'roiwise-state-activations_subject-level_{atlas}_{nrois}.csv')
df = pd.DataFrame(columns=['subject','initial','goal','E_control','error'])

for subj_id in subjects:
    print(f'Starting with subject {subj_id}')
    t = time.time()
    roi_activations = roi_all[roi_all['subject']==subj_id]
    tmp_df=pd.DataFrame([],columns=atlas_order.columns)
    for state1, state2 in itertools.product(states,states):
        roi_tmp = atlas_order.copy()
        roi_tmp['subject'] = [subj_id] * nrois
        roi_tmp['initial'] = [state1] * nrois
        roi_tmp['goal'] = [state2] * nrois
        tmp_df = tmp_df.append(roi_tmp)
    energies, errs = zip(*Parallel(n_jobs=25)(delayed(compute_optimal_energy_roiwise)(roi_activations,state1,state2,A_norm,T,B,rho,S) for state1, state2 in itertools.product(states,states)))
    energies = np.hstack(energies)
    errs = np.hstack(errs)
    tmp_df['E_control'] = energies
    tmp_df['error'] = errs
    df = df.append(tmp_df)
    # df.to_csv(output_dir + f'optimal-transition-energies_roiwise_subject-level_Bmap_{atlas}_{nrois}.csv')
    print(f'Took {time.time()-t:.3f}s')

# df.to_csv(output_dir + f'optimal-transition-energies_roiwise_subject-level_Bmap_{atlas}_{nrois}.csv')

# %% Average control energy

# avg control energy across time
df = pd.read_csv(output_dir + f'optimal-transition-energies_roiwise_subject-level_Bmap_{atlas}_{nrois}.csv')
control_df = pd.DataFrame(columns=atlas_order.columns.to_list()+['E_control','Subject'])
labels = ['AUD', 'CoP', 'CoPar' ,'DMN', 'DAN', 'FrP', 'None', 'RT', 'SAL','SMH','SMM','VAN','VIS','NOTA']
nlabels = len(labels)
id2atl=dict(zip(np.arange(nlabels)+1,labels))

for subj_id in subjects:
    print(f'Starting with subject {subj_id}')
    targets = np.load(dataset_dir + f'{subj_id}_task-rest_bold_desc-raw_labels_{atlas}.npy')
    targets = [*map(id2atl.get,targets)]
    subj_df = df[df.subject==subj_id]
    roi_energy = np.zeros(nrois)
    tpoints = 0
    for ii,current in enumerate(targets):
        if ii==0:
            continue
        previous = targets[ii-1]
        tpoints += 1
        roi_energy += subj_df[(subj_df.initial==previous) & (subj_df.goal==current)]['E_control'].values
    tmp_df = atlas_order.copy()
    tmp_df['subject'] = [subj_id]*nrois
    tmp_df['E_control'] = roi_energy/tpoints
    control_df = control_df.append(tmp_df)
    # control_df.to_csv(output_dir + f'average-optimal-control-energy_roiwise_subject-level_Bmap_{atlas}_{nrois}.csv')
# control_df.to_csv(output_dir + f'average-optimal-control-energy_roiwise_subject-level_Bmap_{atlas}_{nrois}.csv')

# %% Compute ACE by modality

df = pd.read_csv(output_dir + f'optimal-transition-energies_roiwise_subject-level_Bmap_{atlas}_{nrois}.csv')
control_df = pd.DataFrame(columns=atlas_order.columns.to_list()+['E_uu','E_mu','E_um','E_mm','E_sub','subject'])

for subj_id in subjects:
    print(f'Starting with subject {subj_id}')
    targets = np.load(dataset_dir + f'{subj_id}_task-rest_bold_desc-raw_labels_{atlas}.npy')
    targets = [*map(id2atl.get,targets)]
    mods = [*map(lab2mod.get,targets)]
    subj_df = df[df.subject==subj_id]
    uu = np.zeros(nrois)
    um = np.zeros(nrois)
    mu = np.zeros(nrois)
    mm = np.zeros(nrois)
    sub = np.zeros(nrois)
    tuu = 0
    tmu = 0
    tum = 0
    tmm = 0
    tsub = 0
    for ii,current in enumerate(mods):
        if ii==0:
            continue
        previous = mods[ii-1]
        roi_energy = subj_df[(subj_df.initial==previous) & (subj_df.goal==current)]['E_control'].values
        if previous == 'Unimodal' and current == 'Unimodal':
            uu += roi_energy
            tuu += 1 
        elif previous == 'Multimodal' and current == 'Unimodal':
            mu += roi_energy
            tmu += 1
        elif previous == 'Unimodal' and current == 'Multimodal':
            um += roi_energy
            tum += 1
        elif previous == 'Multimodal' and current == 'Multimodal':
            mm += roi_energy
            tmm += 1
        elif current_mod == 'Subthreshold':
            sub += roi_energy
            tsub += 1
        else:
            continue

    tmp_df = atlas_order.copy()
    tmp_df['subject'] = [subj_id]*nrois
    tmp_df['E_uu'] = uu/tuu
    tmp_df['E_um'] = um/tum
    tmp_df['E_mu'] = mu/tmu
    tmp_df['E_mm'] = mm/tmm
    tmp_df['E_sub'] = sub/tsub
    
    control_df = control_df.append(tmp_df)
    # control_df.to_csv(output_dir + f'average-optimal-control-energy_modality_roiwise_subject-level_{atlas}_{nrois}.csv')
# control_df.to_csv(output_dir + f'average-optimal-control-energy_modality_roiwise_subject-level_{atlas}_{nrois}.csv')


