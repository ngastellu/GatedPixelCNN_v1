#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from qcnico.pixel2xyz import pxl2xyz
from qcnico.qcplots import plot_atoms_w_bonds
from qcnico.graph_tools import adjacency_matrix_sparse
from time import perf_counter


def plot_sample(sample_npy, plot_type='pixel'):
    sample_npy = Path(sample_npy)
    if sample_npy.exists():
        structures = np.load(sample_npy)
        for k,s in enumerate(structures):
            # if k > 0: break
            if plot_type == 'pixel':
                plt.imshow(s[0])
            else: # plot XYZ
                print('Starting pixel --> xyz conversion...', end = ' ')
                start = perf_counter()
                pos = pxl2xyz(s[0],pixel2angstroms=0.2)
                end = perf_counter()
                print(f'Done! [{end-start} seconds]')
                rCC = 1.5
                print('Building adjacency matrix..')
                M = adjacency_matrix_sparse(pos, rCC)
                fig, ax = plot_atoms_w_bonds(pos, M, show=False, dotsize=1.0,bond_lw=1.0)
            # plt.suptitle(f'{n} layers #{k} [$T = {T}$]')
            # plt.suptitle(f'{n} {layer_type} layers #{k}')
            plt.suptitle(sample_npy.stem)
            plt.show()
    else:
        print(f'file {sample_npy} not found')



if __name__ == "__main__":
    import sys
    npyfile = Path(sys.argv[1])
    if len(sys.argv) == 3:
        plot_type = sys.argv[2].strip()
    else:
        plot_type = 'pixel'
    plot_sample(npyfile, plot_type)