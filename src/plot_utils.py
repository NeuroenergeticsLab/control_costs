import numpy as np
import matplotlib.pyplot as plt
from nilearn.plotting import plot_surf
from neuromaps.datasets import fetch_atlas
from neuromaps.parcellate import Parcellater
from neuromaps.images import dlabel_to_gifti, annot_to_gifti

def custom_surf_plot(data, space='fsLR', density='32k', template='inflated', cmap='coolwarm', dpi=100,
                     parcellation=None, cbar_label=None, cbar_ticks=None, hemi=None,
                     vmin=None, vmax=None):
    """
    Custom surface plot in fsLR space.
    
    Parameters
    ----------
    data : array_like or tuple
        ROI-wise or vertex-wise data. If tuple, assumes (left, right) 
        hemisphere.
    density : str
        Density of surface plot, can be '8k', '32k' or '164k'.
    template : str
        Type of surface plot. Can be 'inflated', 'veryinflated', 'sphere' or 
        'midthickness' (fsLR, civet) / 'pial' (fsaverage).
    cmap : str
        Colormap.
    dpi : int
        Resolution of plot.
    parcellation : Path or tuple, optional
        Path to an parcellation in .dlabel.nii or .annot format. If tuple, 
        assumes (left, right) GIFTI objects.
    cbar_label: str, optional
        Colorbar label.
    cbar_ticks: list, optional
        Colorbar ticks.
    hemi : str, optional
        Hemisphere to plot. Can be 'left' or 'right'.
    vmin/vmax : int, optional
        Minimun/ maximum value in the plot.
    """
    
    space = space.lower()
    
    if parcellation is not None:
        if not isinstance(parcellation, tuple):
            parcellation = dlabel_to_gifti(parcellation) if space=='fslr' else \
                           annot_to_gifti(parcellation)
        surf_masker = Parcellater(parcellation, space, 
                                  resampling_target='parcellation')
        data = surf_masker.inverse_transform(data)
        l_data, r_data = data[0].agg_data(), data[1].agg_data()
    else:
        if not isinstance(data, tuple):
            raise ValueError("Data input must be tuple of vertex-wise values. \
                             Alternatively, provide 'parcellation' option to use ROI data.")
        l_data, r_data = data[0], data[1]
        
    if None in (vmin, vmax):
        # Handle NaNs in left hemisphere data
        l_min, l_max = np.nanmin(l_data), np.nanmax(l_data)
        l_data = np.nan_to_num(l_data, nan=l_min)
        
        # Handle NaNs in right hemisphere data
        r_min, r_max = np.nanmin(r_data), np.nanmax(r_data)
        r_data = np.nan_to_num(r_data, nan=r_min)
        
        # min/max values in the data
        vmin = np.min([l_min, r_min])
        vmax = np.max([l_max, r_max])
        
    if cbar_ticks is None:
        cbar_ticks = ['min', 'max']
    
    # Fetch surface template for plot
    surfaces = fetch_atlas(space, density)
    lh, rh = surfaces[template]
    
    if hemi == None:
        # Plot both hemispheres
        fig, ax = plt.subplots(nrows=1,ncols=4,subplot_kw={'projection': '3d'}, 
                            figsize=(12, 4), dpi=dpi)
        
        plot_surf(lh, l_data, threshold=-1e-14, cmap=cmap, alpha=1, view='lateral', 
                colorbar=False, axes=ax.flat[0], vmin=vmin, vmax=vmax)
        plot_surf(lh, l_data, threshold=-1e-14, cmap=cmap, alpha=1, view='medial', 
                colorbar=False, axes=ax.flat[1], vmin=vmin, vmax=vmax)

        plot_surf(rh, r_data, threshold=-1e-14, cmap=cmap, alpha=1, view='lateral', 
                colorbar=False, axes=ax.flat[2], vmin=vmin, vmax=vmax)
        p = plot_surf(rh, r_data, threshold=-1e-14, cmap=cmap, alpha=1, view='medial', 
                    colorbar=True, axes=ax.flat[3], vmin=vmin, vmax=vmax)

        p.axes[-1].set_ylabel(cbar_label, fontsize=10, labelpad=0.5)
        p.axes[-1].set_yticks([vmin, vmax])
        p.axes[-1].set_yticklabels(cbar_ticks)
        p.axes[-1].tick_params(labelsize=7, width=0, pad=0.1)
        plt.subplots_adjust(wspace=-0.05)
        p.axes[-1].set_position(p.axes[-1].get_position().translated(0.08, 0))
        
    elif hemi == 'left' or hemi == 'right':
        # Plot one hemisphere
        fig, ax = plt.subplots(nrows=1,ncols=2,subplot_kw={'projection': '3d'}, 
                            figsize=(8, 4), dpi=dpi)
        
        h = lh if hemi == 'left' else rh
        h_data = l_data if hemi == 'left' else r_data
        
        plot_surf(h, h_data, threshold=-1e-14, cmap=cmap, alpha=1, view='lateral', 
                colorbar=False, axes=ax.flat[0], vmin=vmin, vmax=vmax)
        p = plot_surf(h, h_data, threshold=-1e-14, cmap=cmap, alpha=1, view='medial', 
                    colorbar=True, axes=ax.flat[1], vmin=vmin, vmax=vmax)
        
        p.axes[-1].set_ylabel(cbar_label, fontsize=10, labelpad=0.5)
        p.axes[-1].set_yticks([vmin, vmax])
        p.axes[-1].set_yticklabels(cbar_ticks)
        p.axes[-1].tick_params(labelsize=7, width=0, pad=0.1)
        plt.subplots_adjust(wspace=-0.05)
        p.axes[-1].set_position(p.axes[-1].get_position().translated(0.08, 0))