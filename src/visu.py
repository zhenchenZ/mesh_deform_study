import torch
import polyscope as ps
import numpy as np
import torch

import matplotlib.pyplot as plt

class PolyscopeSession:
    """This is a context manager for Polyscope sessions."""
    def __init__(self):
        self._handles = []

    def __enter__(self):
        ps.init()
        return self

    def register(self, handle):
        self._handles.append(handle)
        return handle

    def __exit__(self, exc_type, exc, tb):
        # Remove objects so successive runs don’t stack
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        # Optional: ps.shutdown() if you want to fully close viewer
        # ps.shutdown()
        return False  # don’t suppress exceptions


# def visu_sdf_slice(sdf: np.ndarray, axis: str, slice_ix: int, vmin=None, vmax=None, ax=None):

#     # pick the slice
#     sdf_tensor = torch.from_numpy(sdf).float().cpu() if not torch.is_tensor(sdf) else sdf.float().cpu()
#     axis_idx = {'x':0, 'y':1, 'z':2}[axis]
#     slice = sdf_tensor.index_select(axis_idx, torch.tensor(slice_ix)).squeeze().numpy()

        
#     fig, ax = plt.subplots(1, 2, figsize=(10, 5))

#     # show the sdf values
#     reso = sdf.shape[0]  # assume cubic
#     ax[0].imshow(slice, cmap='jet', vmin=vmin, vmax=vmax)
#     ax[0].set_title(f"SDF Slice at {axis}={slice_ix} ({reso}^3)")
#     fig.colorbar(ax[0].imshow(slice, cmap='jet', vmin=vmin, vmax=vmax), ax=ax[0])

#     # show the signs
#     ax[1].imshow(slice>0, cmap='viridis')
#     ax[1].set_title(f"SDF Sign Slice at {axis}={slice_ix} ({reso}^3)")
    
#     return fig, ax

def visu_sdf_slice_img(
                    sdf: np.ndarray | torch.Tensor, axis: str, slice_ix: int, 
                    vmin=None,
                    vmax=None, 
                    ax=None,
                    figsize=(10, 5),
                ):

    # pick the slice
    sdf_tensor = torch.from_numpy(sdf).float().cpu() if not torch.is_tensor(sdf) else sdf.float().cpu()
    axis_idx = {'x':0, 'y':1, 'z':2}[axis]
    slice = sdf_tensor.index_select(axis_idx, torch.tensor(slice_ix)).squeeze().numpy()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    if vmin is None:
        vmin = slice.min()
    if vmax is None:
        vmax = slice.max()

    # show the sdf values
    reso = sdf.shape[0]  # assume cubic
    ax.imshow(slice, cmap='jet', vmin=vmin, vmax=vmax)
    ax.set_title(f"SDF Slice at {axis}={slice_ix} ({reso}^3)")
    fig = ax.get_figure()
    fig.colorbar(ax.imshow(slice, cmap='jet', vmin=vmin, vmax=vmax), ax=ax)
    
    return ax


def visu_sdf_slice_contour(
        sdf: np.ndarray | torch.Tensor, axis: str, slice_ix: int, 
        log_start=-2,
        log_end=0,
        num_levels=3,
        figsize=(2.75, 2.75),
        dpi=300,
        ax=None,
):
    sdf_tensor = torch.from_numpy(sdf).float().cpu() if not torch.is_tensor(sdf) else sdf.float().cpu()
    axis_idx = {'x':0, 'y':1, 'z':2}[axis]
    array_2d = sdf_tensor.index_select(axis_idx, torch.tensor(slice_ix)).squeeze().numpy()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # build contour levels and colors
    levels_pos = np.logspace(log_start, log_end, num=num_levels)  # logspace, i.e from 10^log_start to 10^log_end
    levels_neg = -1. * levels_pos[::-1]
    levels = np.concatenate((levels_neg, np.zeros((0)), levels_pos), axis=0)
    colors = plt.get_cmap("Spectral")(np.linspace(0., 1., num=num_levels*2+1))

    # plot contours
    CS = ax.contourf(array_2d, levels=levels, colors=colors)
    ax.contour(array_2d, levels=levels, colors='k', linewidths=0.1)
    ax.contour(array_2d, levels=[0], colors='k', linewidths=0.3)

    # show colorbar
    fig = ax.get_figure()
    cbar = fig.colorbar(CS, ax=ax, orientation='vertical')
    ax.axis('off')
    return ax
