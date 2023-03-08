#%% Import libraries
import warnings
warnings.filterwarnings("ignore")
import time
import pyxnat
import os,glob,re,sys,contextlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import nibabel as nib
import seaborn as sns
from scipy.stats import zscore
from nilearn import image,input_data,signal
from pingouin import linear_regression
from joblib import delayed, Parallel
from joblib.externals.loky import set_loky_pickler

#%% Load FSL and ANTs
os.environ["FSLDIR"]='/usr/share/fsl/5.0'
os.environ["FSLOUTPUTTYPE"]='NIFTI_GZ'
os.environ["FSLTCLSH"]='/usr/bin/tclsh'
os.environ["FSLWISH"]='/usr/bin/wish'
os.environ["FSLMULTIFILEQUIT"]='True'
os.environ["LD_LIBRARY_PATH"]='/usr/share/fsl/5.0:/usr/lib/fsl/5.0'

os.environ["ANTSPATH"] = '/opt/ants/bin'
os.environ["PATH"] = f'{os.environ["ANTSPATH"]}:{os.environ["PATH"]}'
if "LD_LIBRARY_PATH" in os.environ:
    os.environ["LD_LIBRARY_PATH"] = f'/opt/ants/lib:{os.environ["LD_LIBRARY_PATH"]}'
else:
    os.environ["LD_LIBRARY_PATH"] = '/opt/ants/lib'

#%% WARNING: If runnning externally, skip this step and instead copy dataset from https://openneuro.org/datasets/ds002898
# Connect to XNAT to retreive validation dataset
project_id = 'Mon_rsPETMR'
project = pyxnat.Interface(config='/home/tumnic/eceballos/xnat_config.cfg').select.project(project_id)

#%% Load functions
def remove_ext(nii_file):
    """
    Remove file extension as in Bash language.
    """
    ext = nii_file.split('.')[-1]
    fn = '.'.join(nii_file.split('.')[:-2]) if ext=='gz' else '.'.join(nii_file.split('.')[:-1])
    return fn
            
def cp_resource(xnat_resource,local_dir,direction,remote_path):
    """
    Copy/upload files from/to XNAT server.
    """
    if (remote_path.find('*') >= 0):
        remote_path = xnat_resource.files(remote_path)[0]._urn
    local_file = os.path.join(local_dir,remote_path)
    if direction == 'remote2local':
        local_path = os.path.dirname(local_file)
        if not os.path.isdir(local_path): 
            !mkdir -p {local_path}
        xnat_resource.file(remote_path).get(local_file)
        return local_file
    elif direction == 'local2remote':
        xnat_resource.file(remote_path).insert(local_file,overwrite=True)
        return None

def subtract_brainstem_cerebellum(subj_id, gm, t1, mni, brainstem_mni, cerebellum_mni, anat_dir):
    """
    Warp brainstem and cerebellum from MNI to native T1 space and subtract from grey matter NIFTI.
    """
    mni2anat = anat_dir + 'mni2anat.mat'
    cerebellum = anat_dir +  f'sub-{subj_id}_cerebellum_2mm_anat.nii.gz'
    brainstem = anat_dir + f'sub-{subj_id}_brainstem_2mm_anat.nii.gz'
    ! flirt -in {mni} -ref {t1} -omat {mni2anat}
    ! flirt -in {cerebellum_mni} -ref {t1} -out {cerebellum} -init {mni2anat} -applyxfm -interp nearestneighbour
    ! flirt -in {brainstem_mni} -ref {t1} -out {brainstem} -init {mni2anat} -applyxfm -interp nearestneighbour
    ! fslmaths {gm} -sub {cerebellum} -sub {brainstem} -bin {gm}

def reg_out_base(time, single_net, all_net_avg, order=3):
    """
    Confound regression
    1)Fit a polynomial to average network signal [all_net_avg].
    -> baseline signal [baseline]
    2)Refit linear model (Ax+b) using baseline as x to optimize the fit to single net signal [single_net].
    -> matrix/vector A and intercept b
    3)Regress out resulting prediction (A * baseline + b) from single net.
    -> regressed signal
    """
    from sklearn import linear_model
    poly_base = np.poly1d(np.polyfit(time, all_net_avg, order))
    baseline = poly_base(time)
    reg = linear_model.LinearRegression().fit(baseline[:,None], single_net)
    single_net_clean = single_net - (reg.predict(baseline[:,None]) + reg.intercept_)# residual = y_true - (X*b_1+b_0) 
    return single_net_clean

def mc_plot(par_file):
    """
    Plot translation and rotation parameters over time.
    """
    par = np.loadtxt(par_file)
    t = np.linspace(0,60,225)
    fig, axes = plt.subplots(2, 1, figsize=(15, 5))
    axes[0].plot(t,par[0:, :3])
    axes[0].set_ylabel('rotation [radians]')
    axes[0].set_ylim(top=0.2,bottom=-0.1)
    axes[1].plot(t,par[0:, 3:])
    axes[1].set_xlabel('time [min]')
    axes[1].set_ylabel('translation [mm]')
    axes[1].set_ylim(top=5.5,bottom=-2.5)
    fig.set_dpi(100);
    

def generate_DCT_regressors(nTRs,TR,HP_cutoff,LP_cutoff):
    """
    Generate cosine waves outside of HP_cutoff and LP_cutoff using Discrete Cosine Transform.
    These waves will later be used for confound regression.
    """
    N=nTRs # timepoints
    K=N # frequency bins
    n = range(N)
    DCT_basis = np.zeros((N, K),dtype=np.float32)
    DCT_basis[:,0] = np.ones((N),dtype=np.float32)/np.sqrt(N)
    
    # DCT-2 as implemented in https://www.mathworks.com/help/signal/ref/dct.html
    for k in range(3,K):
        DCT_basis[:,k] = np.sqrt(2/N)*np.cos([(np.pi/(2*N))*x*(k-1) for x in range(1,N+1)])
    
    # select frequencies to filter out    
    HPC = 1/HP_cutoff
    LPC = 1/LP_cutoff
    nHP = int(np.floor(2*(nTRs*TR)/HPC + 1))
    nLP = int(np.floor(2*(nTRs*TR)/LPC + 1))
    
    # select DCT components to regress out
    to_reg_out = DCT_basis[:,np.concatenate((range(2,nHP),range(int(nLP)-1,nTRs)))]
    
    return to_reg_out

def interpolate(data,tpoints,high_motion_tpoints,TR,nTRs,method='power'):
    """
    Interpolation of censored timepoints either using linear interpolation or as in Power et. al 2014. A frequency transform is used to 
    generate data with the same phase and spectral characteristics as the unflagged data.
    Based on code from rsDenoise (https://github.com/adolphslab/rsDenoise/blob/master/fmriprep_helpers.py)
    
    Parameters
    ----------    
    data: ndarray
        Raw TRs x regions matrix
    tpoints: ndarray
        Timepoints to keep
    high_motion_tpoints: ndarray
        Timepoints to censor out
    TR: int
        Retrieval time
    nTRs: int
        Number of TRs in the data
    method: {'linear','power'}
        Method to perform interpolation
    
    """
    import numpy as np
    tseries = data.copy()
    cens_tseries = tseries[tpoints]
    
    if method == 'linear':
        intpts = np.interp(high_motion_tpoints,tpoints,cens_tseries)
    elif method == 'power':
        N = len(tpoints) # no. of time points
        T = (tpoints.max() - tpoints.min())*TR # total time span
        ofac = 8 # oversampling frequency (generally >=4)
        f_bin = 1/(T*ofac)
        nyquist = 1/(2*TR)

        # compute sampling frequencies
        f = np.arange(0, nyquist, f_bin) + f_bin

        # angular frequencies and constant offsets
        w = 2*np.pi*f
        w = w[:,None]
        t = TR*tpoints[:,None].T
        tau = np.arctan2(np.sum(np.sin(2*w*(t+1)),1),np.sum(np.cos(2*w*(t+1)),1))/(2*np.squeeze(w))

        # compute sampling frequencies
        f = np.arange(f_bin, nyquist+f_bin, f_bin)
        # angular frequencies and constant offsets
        w = 2*np.pi*f
        w = w[:,None]
        t = TR*tpoints[:,None].T
        tau = np.arctan2(np.sum(np.sin(2*w*(t+1)),1),np.sum(np.cos(2*w*(t+1)),1))/(2*np.squeeze(w))

        # spectral power sin and cosine terms (dimensions: nFreqBins x nTimePoints)
        # w * (t_k - tau) = w * t_k - w * tau
        sterm = np.sin(w*(t+1) - (np.squeeze(w)*tau)[:,None])
        cterm = np.cos(w*(t+1) - (np.squeeze(w)*tau)[:,None])

        mean_ct = cens_tseries.mean()
        D = cens_tseries - mean_ct

        c = np.sum(cterm * D,1) / np.sum(np.power(cterm,2),1)
        s = np.sum(sterm * D,1) / np.sum(np.power(sterm,2),1)

        # The inverse function to reconstruct the original time series
        full_tpoints = (np.arange(nTRs)[:,None]+1).T*TR
        prod = full_tpoints*w
        sin_t = np.sin(prod)
        cos_t = np.cos(prod)
        sw_p = sin_t*s[:,None]
        cw_p = cos_t*c[:,None]
        S = np.sum(sw_p,axis=0)
        C = np.sum(cw_p,axis=0)
        H = C + S

        # Normalize the reconstructed spectrum, needed when ofac > 1
        Std_H = np.std(H, ddof=1)
        Std_h = np.std(cens_tseries,ddof=1)
        norm_fac = Std_H/Std_h
        H = H/norm_fac
        H = H + mean_ct

        intpts = H[high_motion_tpoints]
    else:
        raise NameError('Method must be either linear or power.')
        
    tseries[high_motion_tpoints] = intpts
    
    return tseries

def regress_out_confounds(data, confounds):
    """
    Wrapper for linear regression using numpy with subsequent removal of confounds from data.
    Necessary to parallelize using joblib.
    """
    from numpy.linalg import lstsq
    betas = lstsq(confounds, data, rcond=None)[0]
    estimated_data = confounds @ betas
    return data - estimated_data

def clean_signal(data, confounds,TR,HP_cutoff,LP_cutoff):
    """
    Wrapper for linear regression with nilearn with subsequent removal of confounds from data.
    Necessary to parallelize using joblib.
    """
    from nilearn.signal import clean
    residual_ttss = clean(data,confounds=confounds,standardize_confounds=True,standardize='zscore',
                          t_r=TR,filter='butterworth',high_pass=HP_cutoff,low_pass=LP_cutoff)
    return residual_ttss

def supress_stdout(func):
    """
    Supress outputs to avoid excessive number of windows opening.
    """
    def wrapper(*a, **ka):
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                return func(*a, **ka)
    return wrapper

@supress_stdout
def fill_in_missing_regions(data, mask, nrois):
    """
    Check whether data has all ROIs and fill out with zeros if not.
    """
    mask.generate_report()
    roi_numbers = np.arange(nrois) + 1
    rois_in_mask = mask._report_content['summary']['label value']
    missing_rois = np.setdiff1d(roi_numbers, rois_in_mask) - 1 #index starts at 0
    zero_array = np.zeros_like(data[:,0])[:,None]
    
    return np.insert(data,missing_rois,zero_array,axis=1)

#%% General varibales

dataset = 'monash'

if dataset == 'monash':
    sids = np.arange(1,28)
else:
    sids = glob.glob

HP_cutoff = 0.005
LP_cutoff = 0.1
nrois = 400
FD_thr = 0.3
thr = 0.25
thr_i = f'{thr*100:.0f}'
regionwise = True

session = 'petmri'
resource_label = 'fmriprep_v20.2.3'

atlas = 'Gordon'
if atlas is 'Schaefer_400_7N':
    atlas_fn = './data/Schaefer2018_400_7N_MNI152_2mm.nii.gz'
elif atlas is 'Gordon':
    atlas_fn = './data/Gordon2016_333parcels_MNI152_2mm.nii'
    
mni = '/usr/share/fsl/5.0/data/standard/MNI152_T1_2mm_brain.nii.gz'
cerebellum = './data/MNI-maxprob-thr0-2mm_cerebellum.nii.gz'
brainstem = './data/HarvardOxford-sub-maxprob-thr25-2mm_brain-stem.nii.gz'

phys_confounds = ['global_signal',
                  'global_signal_derivative1',
                  'global_signal_derivative1_power2',
                  'global_signal_power2',
                  'csf',
                  'csf_derivative1',
                  'csf_derivative1_power2',
                  'csf_power2',
                  'white_matter',
                  'white_matter_derivative1',
                  'white_matter_derivative1_power2',
                  'white_matter_power2',
                 ]

for sid in sids:
    subj_id = f'{sid:02}'
    print(f'Starting with subject {subj_id}')
    sess_id = subj_id + '-' + session
    cur_sess = project.subject(subj_id).experiment(sess_id)
    
    subj_dir = f'./data/monash/sub-{subj_id}/'
    func_dir = subj_dir + 'func/'
    anat_dir = subj_dir + 'anat/'
    pet_dir = subj_dir + 'pet/'
    if not os.path.isdir(subj_dir):
        os.mkdir(subj_dir)
    if not os.path.isdir(func_dir):
        os.mkdir(func_dir)
    if not os.path.isdir(anat_dir):
        os.mkdir(anat_dir)
    if not os.path.isdir(pet_dir):
        os.mkdir(pet_dir)
    
    t1 = f'sub-{subj_id}_desc-preproc_T1w_brain.nii.gz'
    if not os.path.exists(anat_dir+t1):
        t1_orig = f'sub-{subj_id}_desc-preproc_T1w.nii.gz'
        bm = f'sub-{subj_id}_desc-brain_mask.nii.gz'
        if not os.path.exists(anat_dir+t1_orig):
            project.subject(subj_id).experiment(sess_id).resource(resource_label).file(f'sub-{subj_id}/anat/'+t1_orig).get(anat_dir + t1_orig)
            project.subject(subj_id).experiment(sess_id).resource(resource_label).file(f'sub-{subj_id}/anat/'+bm).get(anat_dir + bm)
        ! fslmaths {anat_dir+t1_orig} -mas {anat_dir+bm} {anat_dir+t1}
    t1 = anat_dir + t1
    
    gm_mask = f'sub-{subj_id}_label-GM_{thr_i}_bin.nii.gz' 
    if not os.path.exists(anat_dir+gm_mask):
        gm_prob = f'sub-{subj_id}_label-GM_probseg.nii.gz'
        project.subject(subj_id).experiment(sess_id).resource(resource_label).file(f'sub-{subj_id}/anat/' + gm_prob).get(anat_dir + gm_prob)
        gm_prob = anat_dir + gm_prob
        ! fslmaths {gm_prob} -thr {thr} -bin -fmedian {anat_dir+gm_mask}
        subtract_brainstem_cerebellum(subj_id,anat_dir+gm_mask,t1,mni,brainstem,cerebellum,anat_dir)
    gm_mask = anat_dir + f'sub-{subj_id}_label-GM_{thr_i}_bin.nii.gz'
    
    all_ttss = []
    all_raw_ttss = []
    run_duration = []
    nruns = 2 if sid == 0 else 6
    TR = 0.72 if sid == 0 else 2.45
    
    for i in np.arange(nruns)+1:
        print(f'\nRun {i}')
        tic = time.time()
        
        # minimally preprocessed run data
        run = f'sub-{subj_id}_task-rest_run-{i}_desc-preproc_bold.nii.gz'
        if not os.path.exists(func_dir+run):
            project.subject(subj_id).experiment(sess_id).resource(resource_label).file(f'sub-{subj_id}/func/'+ run).get(func_dir + run)
        run = func_dir + run
        
        # brain mask
        brain_mask = f'sub-{subj_id}_task-rest_run-{i}_desc-brain_mask.nii.gz'
        if not os.path.exists(anat_dir+brain_mask):
            project.subject(subj_id).experiment(sess_id).resource(resource_label).file(f'sub-{subj_id}/func/'+ brain_mask).get(anat_dir + brain_mask)
        brain_mask = anat_dir + brain_mask
        
        print('Loading data')        
        
        # load ttss
        start = time.time()
        mask = input_data.NiftiMasker(mask_img=brain_mask)
        if not os.path.exists(func_dir+f'sub-{subj_id}-task_rest_run-{i}_raw_vox_ttss.npy'):
            raw_ttss = mask.fit_transform(run)
            np.save(func_dir+f'sub-{subj_id}-task_rest_run-{i}_raw_vox_ttss.npy',raw_ttss)
        else:
            raw_ttss = np.load(func_dir+f'sub-{subj_id}-task_rest_run-{i}_raw_vox_ttss.npy')
            mask = mask.fit(run)
        end = time.time()
        print(f'ttss extraction took {end-start:.3f} seconds')

        # all confounds from fmriprep
        all_confounds = f'sub-{subj_id}_task-rest_run-{i}_desc-confounds_timeseries.tsv'
        if not os.path.exists(func_dir+all_confounds):
            project.subject(subj_id).experiment(sess_id).resource(resource_label).file(f'sub-{subj_id}/func/'+ all_confounds).get(func_dir + all_confounds)
        all_confounds = pd.read_csv(func_dir + all_confounds, delimiter='\t')  
        
        # interpolate TRs with high motion
        nTRs = len(all_confounds)
        regions = raw_ttss.shape[1]
        intp_run = func_dir+f'sub-{subj_id}_task-rest_run-{i}_desc-interpolated_bold.nii.gz'
        high_motion_trs = np.nonzero(all_confounds['framewise_displacement'].to_numpy()>FD_thr)[0]
        tpoints = np.setdiff1d(np.arange(nTRs),high_motion_trs)
        if not os.path.exists(intp_run):
            if sid == 0:
                intp_ttss = Parallel(n_jobs=20)(delayed(interpolate)(raw_ttss[:,region],tpoints,high_motion_trs,TR,nTRs,method='linear') for region in range(regions))
            else:    
                intp_ttss = Parallel(n_jobs=30)(delayed(interpolate)(raw_ttss[:,region],tpoints,high_motion_trs,TR,nTRs,method='power') for region in range(regions))
            intp_ttss = np.array(intp_ttss).T
            np.save(func_dir+f'sub-{subj_id}-task_rest_run-{i}_interpolated_vox_ttss.npy',intp_ttss)
            intp_vol = mask.inverse_transform(intp_ttss)
            intp_vol.to_filename(intp_run)
        intp_ttss = np.load(func_dir+f'sub-{subj_id}-task_rest_run-{i}_interpolated_vox_ttss.npy')
        
        # prepare and regress confounds of interest
        all_confounds[np.isnan(all_confounds)] = 0
#         DCT_components = pd.DataFrame(generate_DCT_regressors(nTRs, TR, HP_cutoff, LP_cutoff))
        confounds = pd.concat([all_confounds[phys_confounds],
#                                all_confounds.filter(like='outlier',axis=1),
                               all_confounds.filter(like='trans',axis=1),
                               all_confounds.filter(like='rot',axis=1)],axis=1)
        confounds_fn = func_dir + f'sub-{subj_id}_task-rest_run-{i}_desc-selected_confounds_timeseries.tsv'
        confounds.to_csv(confounds_fn,sep='\t',index=False,header=False)
        regfilt_run = func_dir + f'sub-{subj_id}_task-rest_run-{i}_desc-regfilt_bold_censored_GSR.nii.gz'
################################################################################################        
        print('Confound regression')        
        if not os.path.exists(regfilt_run):
################################################################################################
            start = time.time()
            if sid==0:
                set_loky_pickler("dill")
                confounds = confounds.to_numpy()
                
                residual_ttss = Parallel(n_jobs=40)(delayed(clean_signal)(intp_ttss[:,region],confounds.to_numpy()) for region in range(regions))
                residual_ttss = np.array(residual_ttss).T
            else:
                residual_ttss = signal.clean(intp_ttss,confounds=confounds,t_r=TR,
                                             high_pass=HP_cutoff,low_pass=LP_cutoff,
                                             standardize_confounds=False,standardize=False)
            end = time.time()
            print(f'Seconds elapsed since regression {end-start:.3f}')
            np.save(func_dir+f'sub-{subj_id}-task_rest_run-{i}_regfilt_censored_GSR_vox_ttss.npy',residual_ttss)
            regfilt_vol = mask.inverse_transform(residual_ttss)
            regfilt_vol.to_filename(regfilt_run)

        # reference run volume
        ref_bold = f'sub-{subj_id}_task-rest_run-{i}_boldref.nii.gz'
        if not os.path.exists(func_dir+ref_bold):
            cur_sess.resource(resource_label).file(f'sub-{subj_id}/func/' + ref_bold).get(func_dir + ref_bold)
        ref_bold = func_dir + f'sub-{subj_id}_task-rest_run-{i}_boldref.nii.gz'
        ref_affine = nib.load(ref_bold).affine
        
        # register GM to func
        gm_func = anat_dir + f'sub-{subj_id}_label-GM_{thr_i}_bin_run-{i}_funcreg.nii.gz'
        print('GM')
        if not os.path.exists(gm_func):    
            anat2func = anat_dir + f'anat2func-run-{i}.txt'
            project.subject(subj_id).experiment(sess_id).resource(resource_label).file(f'sub-{subj_id}/func/sub-{subj_id}_task-rest_run-{i}_from-T1w_to-scanner_mode-image_xfm.txt').get(anat2func)
            ! antsApplyTransforms -i {gm_mask} -r {ref_bold} -o {gm_func} --output-data-type short -n NearestNeighbor -t {anat2func}
            
        # register atlas to func
        atlas_func = anat_dir + f'sub-{subj_id}_{atlas}_rois_run-{i}_funcreg.nii.gz'
        print('Atlas')
        if not os.path.exists(atlas_func):
            mni2anat = anat_dir + 'mni2anat.h5'
            anat2func = anat_dir + f'anat2func-run-{i}.txt'
            project.subject(subj_id).experiment(sess_id).resource(resource_label).file(f'sub-{subj_id}/anat/sub-{subj_id}_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5').get(mni2anat)
            project.subject(subj_id).experiment(sess_id).resource(resource_label).file(f'sub-{subj_id}/func/sub-{subj_id}_task-rest_run-{i}_from-T1w_to-scanner_mode-image_xfm.txt').get(anat2func)
            ! antsApplyTransforms -i {atlas_fn} -r {ref_bold} -o {atlas_func} --output-data-type short -n NearestNeighbor -t {mni2anat} -t {anat2func}
        
        # apply masks to regressed data
        print('Final masking')
        if regionwise:
            mask = input_data.NiftiLabelsMasker(atlas_func,mask_img=gm_func,smoothing_fwhm=6,standardize=True)
        else:
            mask = input_data.NiftiMasker(mask_img=gm_func,smoothing_fwhm=6,standardize=True)
        ttss = mask.fit_transform(regfilt_run)
        raw_ttss = mask.fit_transform(run)
        print(ttss.shape)
        
        # censor high motion trs if there are any
        if not np.array_equal(high_motion_trs,np.empty([0])):
            ttss = np.delete(ttss,high_motion_trs,axis=0)
#         np.save(func_dir+f'sub-{subj_id}-task_rest_bold_run-{i}_rois_ttss.npy',ttss)
        run_duration.append(ttss.shape[0])
#         if (ttss.shape[1] < nrois): ttss = fill_in_missing_regions(ttss, mask, nrois)
        all_ttss.append(ttss)
        all_raw_ttss.append(raw_ttss)
        toc = time.time()
        print(f'Seconds elapsed: {toc-tic:.3f}')
        
    all_ttss = np.vstack(all_ttss)
    all_raw_ttss = np.vstack(all_raw_ttss)
    bold_vol = mask.inverse_transform(all_ttss)
    np.save(func_dir+f'sub-{subj_id}_run-durations.npy',run_duration)
    if regionwise:
        np.save(func_dir+f'sub-{subj_id}-task_rest_bold_{atlas}_rois_ttss_censored_GSR.npy',all_ttss)
        bold_vol.to_filename(func_dir+f'sub-{subj_id}-task_rest_bold_{atlas}_rois_censored_GSR.nii.gz')
        nib.load(atlas_func).to_filename(anat_dir+f'sub-{subj_id}_{atlas}_rois_funcreg.nii.gz')
        nib.load(gm_func).to_filename(anat_dir+f'sub-{subj_id}_label-GM_{thr_i}_bin_funcreg.nii.gz')
    else:
        np.save(func_dir+f'sub-{subj_id}-task_rest_bold_vox_ttss.npy',all_ttss)
        bold_vol.to_filename(func_dir+f'sub-{subj_id}-task_rest_bold_vox.nii.gz')
        nib.load(gm_func).to_filename(anat_dir+f'sub-{subj_id}_label-GM_{thr_i}_bin_funcreg.nii.gz')
    
    
    fig, axs = plt.subplots(nrows=2,ncols=1,dpi=120)
    sns.heatmap(all_ttss.T,cbar=False,yticklabels=False,square=True,cmap='viridis',ax=axs[0])
    sns.heatmap(all_raw_ttss.T,cbar=False,yticklabels=False,square=True,cmap='viridis',ax=axs[1])
    axs[0].set_title(f'Preprocessed | nTRs={len(all_ttss)}')
    axs[1].set_title(f'Raw')
    axs[1].set_xlabel('Time [TR]')
    axs[0].set_ylabel('ROI')
    axs[1].set_ylabel('ROI')
    fig.suptitle(f'sub-{subj_id}')
    fig.tight_layout()
    shift_subplot(axs[1],0.04,-0.05)
    fig.savefig(func_dir+f'sub-{subj_id}-confound_regression_censored_GSR.png')
    print(f'Done with subject {subj_id}\n')   