#%%
import numpy as np
#%%
def phase_randomization(ttss, seed):
    """
    Implementation of phase randomization as in Liegeois et al., 2017, Neuroimage. 
    It preserves the power spectrum of the original signals but randomizes their phase jointly for 
    
    ttss: ndarray
        time series array (time points x voxels/parcels)
    seed: ndarray
        random seed
    
    This was adapted from the original MATLAB code found in https://tinyurl.com/liegeois-pr.
    """
    T, k = ttss.shape
    rng = np.random.RandomState(seed)
    
    # Make the number of samples odd
    if T % 2 == 0:
        T = T - 1
        ttss = ttss[:T, :]
    
    len_ser = (T - 1) // 2
    interv1 = np.arange(1, len_ser+1)
    interv2 = np.arange(len_ser+1, T)
    
    # Fourier transform of the original dataset
    fft_ttss = np.fft.fft(ttss, axis=0) 
    ph_rnd = rng.rand(len_ser)
    
    # Create the random phases for all the time series
    ph_interv1 = np.tile(np.exp(2 * np.pi * 1j * ph_rnd), (k, 1)).T
    ph_interv2 = np.flipud(np.conj(ph_interv1))
    
    # Randomize all the time series simultaneously
    fft_ttss_surr = fft_ttss.copy()
    fft_ttss_surr[interv1, :] = fft_ttss[interv1, :] * ph_interv1
    fft_ttss_surr[interv2, :] = fft_ttss[interv2, :] * ph_interv2
    
    # Inverse transform
    surr = np.real(np.fft.ifft(fft_ttss_surr, axis=0))
    
    return surr