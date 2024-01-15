#%%
import numpy as np
import scipy.io as sio
from scipy.linalg import pinv
from nilearn.plotting import plot_surf
from nibabel.gifti.gifti import GiftiImage
from neuromaps.datasets import fetch_atlas
from matplotlib.pyplot import subplots
from seaborn import color_palette
from pingouin import homoscedasticity, anova, welch_anova, pairwise_tukey, pairwise_gameshowell

def loadmat(filename):
    """
    This function should be called instead of direct scipy.io.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects.
    Taken from https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries
    """
    data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    """
    Checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries.
    """
    for key in dict:
        if isinstance(dict[key], sio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        

def _todict(matobj):
    """
    A recursive function which constructs from matobjects nested dictionaries.
    """
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

def tr_net_labels(df,nlabels,activity_thr):
    """
    Average amplitude network-wise for each time point and assign time point label 
    to network with highest amplitude. If no network surpasses the threshold of
    activity_thr, assign extra label NOTA ('None Of The Above').
    """
    import numpy as np
    net_means = []
    for i in range(1,nlabels):
        # mean across all rois in network
        net_means.append(df[df['network_id']==i]['value'].mean())
    net_means = np.array(net_means)
    idx = np.argmax(net_means)
    if net_means[idx]>=activity_thr:
        return idx + 1
    else:
        return nlabels

def compute_optimal_energy_roiwise(df,state1,state2,A_norm,T,B,rho,S):
    """
    Computes optimal control energy to transition from state1 to state2.
    """
    import numpy as np
    from network_control.energies import optimal_input
    from network_control.energies import integrate_u
    from scipy import integrate
    nrois = len(A_norm)
    initial = df[df['state'] == state1]['beta'].values[:,None]
    goal = df[df['state'] == state2]['beta'].values[:,None]
    if np.array_equal(initial,np.empty([0,1])) or np.array_equal(goal,np.empty([0,1])):
        energy, n_err = np.repeat(np.nan,nrois),np.repeat(np.nan,nrois)
    else:
        x, u, n_err = optimal_input(A_norm, T, B, initial, goal, rho, S)
        energy = integrate_u(u)
        n_err = np.repeat(n_err,nrois)
    return energy, n_err

def compute_optimal_energy_roiwise_array(state1,state2,A_norm,T,B,rho,S):
    """
    Computes optimal control energy to transition from state1 to state2.
    States are passed as numpy arrays.
    """
    import numpy as np
    from network_control.energies import optimal_input
    from network_control.energies import integrate_u
    from scipy import integrate
    nrois = len(A_norm)
    x, u, n_err = optimal_input(A_norm, T, B, state1, state2, rho, S)
    energy = integrate_u(u)
    n_err = np.repeat(n_err,nrois)
    return energy, n_err

def compute_optimal_energy_roiwise_null(df,state1,state2,A_norm,T,B,rho,S):
    """
    Computes optimal control energy to transition from state1 to state2.
    """
    import numpy as np
    from network_control.energies import optimal_input
    from network_control.energies import integrate_u
    from scipy import integrate
    nrois = len(A_norm)
    initial = df[df['state'] == state1]['beta'].values[:,None]
    goal = df[df['state'] == state2]['beta'].values[:,None]
    if np.array_equal(initial,np.empty([0,1])) or np.array_equal(goal,np.empty([0,1])):
        print(f"Subject has no {state1} to {state2} transition")
        energy = np.repeat(np.nan,nrois)
    else:
        x, u, n_err = optimal_input(A_norm, T, B, initial, goal, rho, S)
        energy = integrate_u(u)
    return energy.astype(int)

def compute_optimal_energy_roiwise_slurm(betas, state1, state2, A_norm, T, B, rho, S):
    """
    Revised version of 'compute_optimal_energy_roiwise' to run more efficiently in SLURM. 
    Computes optimal control energy to transition from state1 to state2.
    """

    import numpy as np
    from network_control.energies import optimal_input
    from network_control.energies import integrate_u
    from scipy import integrate
    nrois = len(A_norm)
    initial = betas[state1]
    goal = betas[state2]
    if np.array_equal(initial, np.zeros(nrois)) or np.array_equal(goal, np.zeros(nrois)):
        print(f"Subject has no {state1} to {state2} transition")
        energy = np.repeat(np.nan, nrois)
        return energy
    else:
        x, u, n_err = optimal_input(A_norm, T, B, initial, goal, rho, S)
        energy = integrate_u(u)
        return energy.astype(int)

def plot_surface(left, right, template, density, surf='inflated', data_dir=None, cmap='inferno', 
                 vmin=None, vmax=None, fig_title=None, cbar_label=None, dpi=100):

    if all(isinstance(elem, GiftiImage) for elem in (left, right)):
        l_map = left.agg_data()
        r_map = right.agg_data()
    
    elif all(isinstance(elem, (str, np.ndarray, np.generic)) for elem in (left, right)):
        l_map = left
        r_map = right
    else:
        raise ValueError("Map inputs 'left' and 'right' have to be either str, ndarray or GiftiImage.")
    
    atlas = fetch_atlas(template, density, data_dir=data_dir, verbose=0)
    
    fig, ax = subplots(nrows=2, ncols=2, subplot_kw={'projection': '3d'}, dpi=dpi)
    
    # Left lateral
    plot_surf(str(atlas[surf][0]), l_map, hemi='left', 
                    colorbar=False, cbar_tick_format='%.0f', cmap=cmap, 
                    axes=ax.flat[0], vmin=vmin, vmax=vmax)
    # Left medial
    plot_surf(str(atlas[surf][0]), l_map, hemi='left', view='medial',
                    colorbar=False, cbar_tick_format='%.0f', cmap=cmap, 
                    axes=ax.flat[1], vmin=vmin, vmax=vmax)
    # Right lateral
    plot_surf(str(atlas[surf][1]), r_map, hemi='right', 
                    colorbar=False, cbar_tick_format='%.0f', cmap=cmap, 
                    axes=ax.flat[2], vmin=vmin, vmax=vmax)
    # Right medial
    p = plot_surf(str(atlas[surf][1]), r_map, hemi='right', view='medial',
                    colorbar=True, cbar_tick_format='%.0f', cmap=cmap, 
                    axes=ax.flat[3], vmin=vmin, vmax=vmax)
    p.axes[-1].set_ylabel(cbar_label,fontsize=7);
    fig.suptitle(fig_title)

def phase_randomization_cov(ttss, seed):
    """
    Implementation of phase randomization as in Liegeois et al., 2017, Neuroimage. 
    It preserves the power spectrum of the original signals but randomizes their phase 
    jointly to preserve the original covariance
    
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
    
    # Create the random phases for all the time series jointly
    ph_interv1 = np.tile(np.exp(2 * np.pi * 1j * ph_rnd), (k, 1)).T
    ph_interv2 = np.flipud(np.conj(ph_interv1))
    
    # Randomize all the time series simultaneously
    fft_ttss_surr = fft_ttss.copy()
    fft_ttss_surr[interv1, :] = fft_ttss[interv1, :] * ph_interv1
    fft_ttss_surr[interv2, :] = fft_ttss[interv2, :] * ph_interv2
    
    # Inverse transform
    surr = np.real(np.fft.ifft(fft_ttss_surr, axis=0))
    return surr

def phase_randomization_nocov(ttss, seed):
    """
    Modification of phase randomization from Liegeois et al., 2017, Neuroimage. 
    It preserves the power spectrum of the original signals but randomizes phase
    of signals individually to break the original covariance.

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
    ph_rnd = rng.rand(len_ser, k)
    
    # Create the random phases for all the time series individually
    ph_interv1 = np.exp(2 * np.pi * 1j * ph_rnd)
    ph_interv2 = np.flipud(np.conj(ph_interv1))
    
    # Randomize all the time series simultaneously
    fft_ttss_surr = fft_ttss.copy()
    fft_ttss_surr[interv1, :] = fft_ttss[interv1, :] * ph_interv1
    fft_ttss_surr[interv2, :] = fft_ttss[interv2, :] * ph_interv2
    
    # Inverse transform
    surr = np.real(np.fft.ifft(fft_ttss_surr, axis=0))
    return surr

def strength_preserving_rand(A, rewiring_iter = 10, nstage = 100, niter = 10000,
                             temp = 1000, frac = 0.5,
                             energy_func = None, energy_type = 'euclidean',
                             connected = None, verbose = False, seed = None):

    """
    Degree- and strength-preserving randomization of
    undirected, weighted adjacency matrix A
    Parameters
    ----------
    A : (N, N) array-like
        Undirected symmetric weighted adjacency matrix
    rewiring_iter : int, optional
        Rewiring parameter (each edge is rewired approximately maxswap times).
        Default = 10.
    nstage : int, optional
        Number of annealing stages. Default = 100.
    niter : int, optional
        Number of iterations per stage. Default = 10000.
    temp : float, optional
        Initial temperature. Default = 1000.
    frac : float, optional
        Fractional decrease in temperature per stage. Default = 0.5.
    energy_type: str, optional
        Energy function to minimize. Can be either:
            'euclidean': Euclidean distance between strength sequence vectors
                         of the original network and the randomized network
            'max': The single largest value
                   by which the strength sequences deviate
            'mae': Mean absolute error
            'mse': Mean squared error
            'rmse': Root mean squared error
        Default = 'euclidean'.
    energy_func: callable, optional
        Callable with two positional arguments corresponding to
        two strength sequence numpy arrays that returns an energy value.
        Overwrites “energy_type”.
        See “energy_type” for specifying a predefined energy type instead.
    connected: bool, optional
        Maintain connectedness of randomized network.
        By default, this is inferred from data.
    verbose: bool, optional
        Print status to screen at the end of every stage. Default = False.
    seed: float, optional
        Random seed. Default = None.
    Returns
    -------
    B : (N, N) array-like
        Randomized adjacency matrix
    min_energy : float
        Minimum energy obtained by annealing
    Notes
    -------
    Uses Maslov & Sneppen rewiring model to produce a
    surrogate adjacency matrix, B, with the same size, density, degree sequence,
    and weight distribution as A. The weights are then permuted to optimize the
    match between the strength sequences of A and B using simulated annealing.
    References
    -------
    Misic, B. et al. (2015) Cooperative and Competitive Spreading Dynamics
    on the Human Connectome. Neuron.
    2014-2022
    Richard Betzel, Indiana University
    Filip Milisav, McGill University
    Modification History:
    2014: Original (Richard Betzel)
    2022: Python translation, added connectedness-preservation functionality,
          new predefined energy types, and
          user-provided energy callable functionality (Filip Milisav)
    """
    import bct
    try:
        A = np.array(A)
    except ValueError as err:
        msg = ('A must be array_like. Received: {}.'.format(type(A)))
        raise TypeError(msg) from err

    rs = np.random.RandomState(seed)

    n = A.shape[0]
    s = np.sum(A, axis = 1) #strengths of A

    if connected is None:
        connected = False if bct.number_of_components(A) > 1 else True

    #Maslov & Sneppen rewiring
    if connected:
        B = bct.randmio_und_connected(A, rewiring_iter, seed = seed)[0]
    else:
        B = bct.randmio_und(A, rewiring_iter, seed = seed)[0]

    u, v = np.triu(B, k = 1).nonzero() #upper triangle indices
    wts = np.triu(B, k = 1)[(u, v)] #upper triangle values
    m = len(wts)
    sb = np.sum(B, axis = 1) #strengths of B

    if energy_func is not None:
        energy = energy_func(s, sb)
    elif energy_type == 'euclidean':
        energy = np.sum((s - sb)**2)
    elif energy_type == 'max':
        energy = np.max(np.abs(s - sb))
    elif energy_type == 'mae':
        energy = np.mean(np.abs(s - sb))
    elif energy_type == 'mse':
        energy = np.mean((s - sb)**2)
    elif energy_type == 'rmse':
        energy = np.sqrt(np.mean((s - sb)**2))
    else:
        msg = ("energy_type must be one of 'euclidean', 'max', "
               "'mae', 'mse', or 'rmse'. Received: {}.".format(energy_type))
        raise ValueError(msg)

    energymin = energy
    wtsmin = wts

    if verbose:
        print('\ninitial energy {:.5f}'.format(energy))

    for istage in range(nstage):

        naccept = 0
        for i in range(niter):

            #permutation
            e1 = rs.randint(m)
            e2 = rs.randint(m)

            a, b = u[e1], v[e1]
            c, d = u[e2], v[e2]

            sb_prime = sb.copy()
            sb_prime[[a, b]] = sb_prime[[a, b]] - wts[e1] + wts[e2]
            sb_prime[[c, d]] = sb_prime[[c, d]] + wts[e1] - wts[e2]

            if energy_func is not None:
                energy_prime = energy_func(sb_prime, s)
            elif energy_type == 'euclidean':
                energy_prime = np.sum((sb_prime - s)**2)
            elif energy_type == 'max':
                energy_prime = np.max(np.abs(sb_prime - s))
            elif energy_type == 'mae':
                energy_prime = np.mean(np.abs(sb_prime - s))
            elif energy_type == 'mse':
                energy_prime = np.mean((sb_prime - s)**2)
            elif energy_type == 'rmse':
                energy_prime = np.sqrt(np.mean((sb_prime - s)**2))
            else:
                msg = ("energy_type must be one of 'euclidean', 'max', "
                       "'mae', 'mse', or 'rmse'. "
                       "Received: {}.".format(energy_type))
                raise ValueError(msg)

            #permutation acceptance criterion
            if (energy_prime < energy or
               rs.rand() < np.exp(-(energy_prime - energy)/temp)):
                sb = sb_prime.copy()
                wts[[e1, e2]] = wts[[e2, e1]]
                energy = energy_prime
                if energy < energymin:
                    energymin = energy
                    wtsmin = wts
                naccept = naccept + 1

        #temperature update
        temp = temp*frac
        if verbose:
            print('\nstage {:d}, temp {:.5f}, best energy {:.5f}, '
                  'frac of accepted moves {:.3f}'.format(istage, temp,
                                                         energymin,
                                                         naccept/niter))

    B = np.zeros((n, n))
    B[(u, v)] = wtsmin
    B = B + B.T

    return B, energymin

def network_variance(node_distribution, network_structure, use_given_YN=False):
    # cfr Devriendt,... Lambiotte 2020, 2022
    # the network must be connected
    # The distribution must be positive and sum to 1
    
    if not use_given_YN:
        use_given_YN = False
    
    # only positive values allowed; must sum up to 1
    if np.min(node_distribution) < 0:
        node_distribution = node_distribution - np.min(node_distribution)
    
    if np.sum(node_distribution) != 1:
        node_distribution = node_distribution / np.sum(node_distribution)
    
    p = node_distribution.reshape(-1, 1)
    if p.shape[0] == 1:
        p = p.T
    
    if not use_given_YN:
        # Use effective resistance for the distance between each pair of nodes,
        # based on all paths between them
        Ohm = effective_resistance(network_structure)
    
    elif use_given_YN:
        # eg when using Euclidean
        Ohm = network_structure
    
    net_var = 0.5 * np.dot(np.dot(p.T, Ohm), p)
    return net_var
    

def effective_resistance(mat):
    # Distance metric that takes into account all paths, not just shortest ones;
    # So the more paths, the smaller the resistance
    # cfr Eq 21. https://nas.ewi.tudelft.nl/people/Piet/papers/LAA_2011_EffectiveResistance.pdf
    
    n = mat.shape[0]  # number of vertices
    mat = mat / np.max(mat)
    
    D = np.diag(np.sum(mat, axis=0))
    L = D - mat  # Laplacian
    
    Qstar = pinv(L)  # Moore-Penrose pseudoinverse
    
    W = np.zeros((n, n))
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = 1
        
        for j in range(n):
            ej = np.zeros(n)
            ej[j] = 1
            
            W[i, j] = ((ei - ej).T @ Qstar) @ (ei - ej)
    
    return W

def get_gordon_palette(color_id = None):
    cmap = np.array([[130.,102.,168.,256],
                [94.,69.,142.,256],
                [205.,208.,175.,256],
                [200.,88.,74.,256],
                [133.,185.,102.,256],
                [232.,230.,118.,256],
                [255.,255.,255.,256],
                [191.,158.,189.,256],
                [40.,40.,40.,200],
                [162.,206.,217.,256],
                [225.,152.,88.,256],
                [95.,143.,147.,256],
                [60.,62.,127.,256],
                [235.,237.,236.,256]])/256
    
    if (color_id is None):
        color_id = np.arange(len(cmap))
        
    cmap = cmap[color_id]
    
    return color_palette(cmap,n_colors=len(cmap))

def test_multigroup_mean(df, ntests=1, dv=None, group=None):
    # if equal variances
    if homoscedasticity(df, dv=dv, group=group)['equal_var'].values == True:
        
        p_bonf = anova(df, dv=dv, between=group)['p-unc'].values[0] * ntests
        if p_bonf <= 0.05:
            # post-hoc pairwise test
            return print(pairwise_tukey(data=df, dv=dv, between=group))
        else:
            return print(f'Corrected ANOVA p-value of {p_bonf:.03f} was not significant.')
    # if unequal variances
    else:
        p_bonf = welch_anova(df, dv=dv, between=group)['p-unc'].values * ntests
        if p_bonf <= 0.05:
            # post-hoc pairwise test
            return print(pairwise_gameshowell(data=df, dv=dv, between=group))
        else:
            return print(f'Corrected Welch-ANOVA p-value of {p_bonf:.03f} was not significant.')

def get_significance_string(p, type='asterisk'):
    if p>0.05:
        return 'ns'
    elif p<0.001:
        return '***' if type=='asterisk' else 'p < 0.001'
    elif p<0.01:
        return '**' if type=='asterisk' else 'p < 0.01'
    elif p<=0.05:
        return '*' if type=='asterisk' else 'p ≤ 0.05'

def plot_network(A, coords, edge_scores, node_scores, edge_cmap="Greys",
                 node_cmap="viridis", edge_alpha=0.25, node_alpha=1,
                 edge_vmin=None, edge_vmax=None, node_vmin=None,
                 node_vmax=None, nodes_color='black', edges_color='black',
                 linewidth=0.25, s=100, view_edge=True, figsize=None):
    '''
    Function to draw (plot) a network of nodes and edges.
    Parameters
    ----------
    A : (n, n) ndarray
        Array storing the adjacency matrix of the network. 'n' is the
        number of nodes in the network.
    coords : (n, 3) ndarray
        Coordinates of the network's nodes.
    edge_scores: (n,n) ndarray
        Array storing edge scores for individual edges in the network. These
        scores are used to color the edges.
    node_scores : (n) ndarray
        Array storing node scores for individual nodes in the network. These
        scores are used to color the nodes.
    edge_cmap, node_cmap: str
        Colormaps from matplotlib.
    edge_alpha, node_alpha: float, optional
        The alpha blending value, between 0 (transparent) and 1 (opaque)
    edge_vmin, edge_vmax, node_vmin, node_vmax: float, optional
        Minimal and maximal values of the node and edge colors. If None,
        the min and max of edge_scores and node_scores respectively are used.
        Default: `None`
    nodes_color, edges_color: str
        Color to be used to plot the network's nodes and edges if edge_scores
        or node_scores are none.
    linewidth: float
        Width of the edges.
    s: float or array-like
        Size the nodes.
    view_edge: bool
        If true, network edges are shown.
    figsize: (float, float)
        Width and height of the figure, in inches.
    Returns
    -------
    fig: matplotlib.figure.Figure instance
        Figure instance of the drawn network.
    ax: matplotlib.axes.Axes instance
        Ax instance of the drawn network.
    '''
    from matplotlib import cm, colors
    if figsize is None:
        figsize = (10, 10)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    # Plot the edges
    if view_edge:
        # Identify edges in the network
        edges = np.where(A > 0)
        # Get the color of the edges
        if edge_scores is None:
            edge_colors = np.full((len(edges[0])), edges_color, dtype="<U10")
        else:
            edge_colors = cm.get_cmap(edge_cmap)(
                colors.Normalize(edge_vmin, edge_vmax)(edge_scores[edges]))
        # Plot the edges
        for edge_i, edge_j, c in zip(edges[0], edges[1], edge_colors):
            x1, x2 = coords[edge_i, 0], coords[edge_j, 0]
            y1, y2 = coords[edge_i, 1], coords[edge_j, 1]
            ax.plot([x1, x2], [y1, y2], c=c, linewidth=linewidth,
                    alpha=edge_alpha, zorder=0)
    # Get the color of the nodes
    if node_scores is None:
        node_scores = nodes_color
    node_colors = node_scores
    # plot the nodes
    ax.scatter(
        coords[:, 0], coords[:, 1], c=node_colors,
        edgecolors='none', cmap=node_cmap, vmin=node_vmin,
        vmax=node_vmax, alpha=node_alpha, s=s, zorder=1, edgecolor='k')
    ax.set_aspect('equal')
    ax.axis('off')
    return fig, ax

def partial_corr(X, Y, Z):
    # check that X, Y, Z are all 2D arrays
    if X.ndim == 1:
        X = X[:, np.newaxis]
    if Y.ndim == 1:
        Y = Y[:, np.newaxis]
    if Z.ndim == 1:
        Z = Z[:, np.newaxis]
    
    # stack variables into data array
    data = np.hstack([X, Y, Z])
    
    V = np.cov(data, rowvar=False)
    Vi = np.linalg.pinv(V, hermitian=True)  # inverse covariance matrix
    Vi_diag = Vi.diagonal()
    D = np.diag(np.sqrt(1 / Vi_diag))
    pcor = -1 * (D @ Vi @ D)  # partial correlation matrix
    
    return pcor[0,1]

def pcorr_significance(src, trg, covariates, nulls):
    r_true = partial_corr(src, trg, covariates)
    n_perm = nulls.shape[-1]
    permuted_results = []

    for perm in range(n_perm):
        src_perm = src[nulls[:, perm]]
        permuted_results.append(partial_corr(src_perm, trg, covariates))

    p = (1 + sum(np.abs(permuted_results) > np.abs(r_true))) / (1 + n_perm)
    
    return r_true, p