#%%
import matplotlib.colors as mpc
import numpy as np
import seaborn as sns
from matplotlib.cm import ColormapRegistry

def sequential_blue(N=100, return_palette=False, n_colors=8):
    # taken from https://coolors.co/f4f5f5-e8eaed-bed5e1-93bfd5-2b7ea1-2b6178
    # clist = ['d2d6da','f5f5f5', 'e8eaed', 'bed5e1', '93bfd5', '2b7ea1', '2b6178']
    clist = ["d2d6da","e8eaed","bed5e1","93bfd5","2b7ea1","2b6178"]
    hex = [f'#{c}' for c in clist]
    rgb = list(map(mpc.to_rgb, hex))
    if return_palette:
        return sns.color_palette(rgb, n_colors=n_colors)
    else:
        return mpc.LinearSegmentedColormap.from_list('custom', rgb, N=N)

def sequential_green(N=100, return_palette=False, n_colors=8):
    clist = ["e7f0ee","c4dcd2","a4c5b8","79aa94","4c8a70","206246","114d33","013721"]
    hex = [f'#{c}' for c in clist]
    rgb = list(map(mpc.to_rgb, hex))
    if return_palette:
        return sns.color_palette(rgb, n_colors=n_colors)
    else:
        return mpc.LinearSegmentedColormap.from_list('custom', rgb, N=N)

def custom_coolwarm(N=100, return_palette=False, n_colors=8):
    # clist = ["20495a","2e86ab","93bfd5","d2d6da","f2a9a3","eb5a47","d65241"]
    # clist = ["2a6179","3d758d","75aec7","c2d4dc","ebebeb","e5d3d1","ea9085","c86356","b73a2a"]
    clist = ["2a6179","3d758d","75aec7","c2d4dc","ebebeb","e5d3d1","ea9085","c86356","b73a2a"]
    hex = [f'#{c}' for c in clist]
    rgb = list(map(mpc.to_rgb, hex))
    if return_palette:
        return sns.color_palette(rgb, n_colors=n_colors)
    else:
        return mpc.LinearSegmentedColormap.from_list('custom', rgb, N=N)

def categorical_cmap(N=None, return_palette=False, n_colors=8):
    clist = ["ea6b5d","65a488","498eab"]
    hex = [f'#{c}' for c in clist]
    rgb = list(map(mpc.to_rgb, hex))
    N = len(rgb) if N==None else N
    if return_palette:
        return sns.color_palette(rgb, n_colors=n_colors)
    else:
        return mpc.LinearSegmentedColormap.from_list('custom', rgb, N=N)
    

def cmap_from_hex(clist, N=100, return_palette=False, n_colors=8):
    hex = [f'#{c}' for c in clist]
    rgb = list(map(mpc.to_rgb, hex_list))
    if return_palette:
        return sns.color_palette(rgb, n_colors=n_colors)
    else:
        return mpc.LinearSegmentedColormap.from_list('custom', rgb, N=N)

def schaefer_cmap(include_nota = False):
    if include_nota:
        rgb = np.array([(119, 17, 128), # Vis
                        (70, 128, 179), # SomMot
                        (4, 117, 14), # DorsAttn
                        (200, 56, 246), # SalVentAttn
                        (223, 249, 163), # Limbic
                        (232, 147, 31), # Cont
                        (218, 24, 24), # Default
                        (255, 255, 255) # None of the above
                        ]) / 255
    else:
        rgb = np.array([(119, 17, 128), # Vis
                        (70, 128, 179), # SomMot
                        (4, 117, 14), # DorsAttn
                        (200, 56, 246), # SalVentAttn
                        (223, 249, 163), # Limbic
                        (232, 147, 31), # Cont
                        (218, 24, 24) # Default
                        ]) / 255
    return mpc.LinearSegmentedColormap.from_list('custom', rgb, N=len(rgb))