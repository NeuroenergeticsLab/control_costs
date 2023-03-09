# %%
import warnings
warnings.filterwarnings("ignore")
import glob, time, itertools
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy as sp
from joblib import Parallel,delayed
from nilearn.input_data import NiftiLabelsMasker


# import os
# import itertools
# import pyxnat
# import numpy as np 
# import pandas as pd
# import nibabel as nib
# import scipy as sp
# import matplotlib.pyplot as plt
# import seaborn as sns
# import networkx as nx

# from dyneusr import DyNeuGraph
# from nilearn.input_data import NiftiMasker, NiftiLabelsMasker
# from nilearn.image import threshold_img, binarize_img, math_img # index_img,mean_img
# from nilearn.plotting import plot_stat_map
# from kmapper import KeplerMapper, Cover
# from sklearn.manifold import TSNE
# from sklearn.cluster import DBSCAN
# from pingouin import linear_regression

# os.environ["FSLDIR"]='/usr/share/fsl/5.0'
# os.environ["FSLOUTPUTTYPE"]='NIFTI_GZ'
# os.environ["FSLTCLSH"]='/usr/bin/tclsh'
# os.environ["FSLWISH"]='/usr/bin/wish'
# os.environ["FSLMULTIFILEQUIT"]="True"
# os.environ["LD_LIBRARY_PATH"]='/usr/share/fsl/5.0:/usr/lib/fsl/5.0'

# %%
def tr_net_labels(df,nlabels,activity_thr):
    """
    Average amplitude network-wise for each time point and assign time point label 
    to network with highest amplitude. If no network surpasses the threshold of
    activity_thr, assign extra label NOTA ('None Of The Above').
    """
    import numpy as np
    net_means = []
    for i in range(2,nlabels):
        # mean across all rois in network
        net_means.append(df[df['network_id']==i]['value'].mean())
    net_means = np.array(net_means)
    idx = np.argmax(net_means)
    if net_means[idx]>=0.5:
        return idx+2
    else:
        return nlabels

# %%
subjects = sorted([fn.split('-')[1][:5] for fn in glob.glob('./data/msc/sub*even.npy')])

# %% [markdown]
# ### General variables

# %%
dataset = 'hcp'
root_dir = './data/'
dataset_dir = root_dir + f'{dataset}/'
output_dir = f'./results/{dataset}/'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    ########## os.mkdir(output_dir + 'state_maps/')

if dataset == 'hcp':
    subjects = sorted([fn.split('/')[-1][:10] for fn in glob.glob(dataset_dir + 'sub*timeseries*.npy')])
elif dataset == 'monash':
    subjects = [f'sub-{sid:02}' for sid in np.arange(1,28)]

atlas = 'Gordon'
if atlas == 'Gordon':
    atlas_fn = root_dir + 'Gordon2016_space-MNI152_den-3mm.nii'
    labels = ['AUD', 'CoP', 'CoPar' ,'DMN', 'DAN', 'FrP', 'None', 'RT', 'SAL','SMH','SMM','VAN','VIS','NOTA']
    atlas_order = pd.read_csv(root_dir + 'gordon2016_parcels.csv')
    atlas_order = atlas_order.drop(columns=[atlas_order.columns[-1],'Surface area (mm2)','Centroid (MNI)'])
    atl2id = dict(zip(labels,np.arange(nlabels)+1))
    atl_id2label = dict(zip(atlas_order['ParcelID'].tolist(), atlas_order['Community']))
    atlas_order['CommunityID'] = atlas_order['Community'].map(atl2id)
    
elif atlas == 'Schaefer':
    atlas_fn = root_dir + 'Schaefer2018_400_7N_space-MNI152_den-2mm.nii.gz'
    labels = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default']
    atlas_order = pd.read_csv(root_dir + 'Schaefer2018_400_7N_order.txt',sep='\t',header=None)
    atlas_order['network'] = atlas_order[1].str.split('_').str.get(2)
    atl2id = dict(zip(atlas_order['network'].unique(),range(1,N+1)))
    atl_id2label = dict(zip(atlas_order[0].tolist(), atlas_order['network']))
    atlas_order['network_id'] = atlas_order['network'].map(atl2id)
    
nlabels = len(labels)
id2atl = dict(zip(np.arange(nlabels)+1,labels))
states = labels
nstates = len(states)
############ nrois = len(atlas_order)
activity_thr = 0.5

# DataFrame to store all results
roi_activations = pd.DataFrame(columns=['subject','state','T','beta'])

# %%
for subj_id in subjects:
    print(f'Starting with subject {subj_id}')
    subj_dir = dataset_dir + f'{subj_id}/'
    func_dir = subj_dir + 'func/'
    ###### ttss = np.load(func_dir + f'sub-{subj_id}-task_rest_bold_Gordon_rois_ttss_censored_GSR.npy')
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
    ttss_df['network'] = ttss_df['roi'].map(atl_id2label)
    ttss_df['network_id'] = ttss_df['network'].map(atl2id)

    # for each TR calculate the network with the most (positively) active ROIs
    targets = Parallel(n_jobs=30)(delayed(tr_net_labels)(ttss_df[ttss_df.tr==tr],nlabels,activity_thr) for tr in range(len(ttss)))
    targets = np.array(targets)

    np.save(dataset_dir + f'{subj_id}_task-rest_bold_desc-raw_labels_{atlas}.npy',targets)
    
    # one-hot-encode states
    states = pd.DataFrame({lab:(targets==lab_idx+1).astype(int) for lab_idx,lab in enumerate(labels)}).values

    for i,state in enumerate(states.T):
        if state.sum()==0:
            continue
        print(f'State: {labels[i]}')
        Tvalues = []
        betas = []
        for roi in range(ttss.shape[1]):
            model = linear_regression(state,ttss[:,roi])
            Tvalues.append(model['T'][1])
            betas.append(model['coef'][1])
        Tvalues = np.array(Tvalues)
        betas = np.array(betas)

        ###### Tmap = mask.inverse_transform(Tvalues[None,:])
        # Tmap.to_filename(output_dir + f'state_maps/{subj_id}_desc-Tmap-{labels[j]}_space-MNI152_Gordon_333.nii.gz')
        # Bmap = mask.inverse_transform(betas[None,:])
        # Bmap.to_filename(output_dir + f'state_maps/{subj_id}_desc-Bmap-{labels[j]}_space-MNI152_Gordon_333.nii.gz')

        tmp_df = atlas_order.copy()
        tmp_df['subject'] = [f'{subj_id}'] * nrois
        tmp_df['state'] = [labels[i]] * nrois
        tmp_df['T'] = Tvalues
        tmp_df['beta'] = betas
        roi_activations = roi_activations.append(tmp_df)
        roi_activations.to_csv(output_dir + f'roiwise-state-activations_subject-level_{atlas}_{nrois}.csv')
        
roi_activations.to_csv(output_dir + f'roiwise-state-activations_subject-level_{atlas}_{nrois}.csv')


