from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
import os
base_path = "./graphs/"
fileset = Path(base_path).glob('*.indptr')

num_warps = 12
warp_max_nz = 64
deg_bound = num_warps * warp_max_nz

meta_dir = './w' + str(num_warps) + '_nz' + str(warp_max_nz) + '_warp_4/'
if not os.path.exists(meta_dir):
    os.makedirs(meta_dir)
print(f"generating metadata, save to {meta_dir}")

fcnt = 0
for file in fileset:
    fcnt += 1
    indptr = np.fromfile(base_path + file.stem + ".indptr", dtype=np.int32)
    indices = np.fromfile(base_path + file.stem + ".indices", dtype=np.int32)
    v_num = len(indptr) - 1
    e_num = len(indices)
    vals = np.ones(e_num)
    csr = csr_matrix((vals, indices, indptr))
    warp_row = []
    warp_loc = []
    warp_len = []
    cur_loc = 0
    for i in range(v_num):
        cur_degree = indptr[i+1] - indptr[i]
        if cur_degree == 0:
            continue
        tmp_loc = 0
        while True:
            warp_row.append(i)
            warp_loc.append(cur_loc)
            if cur_degree - tmp_loc <= warp_max_nz:
                warp_len.append(cur_degree - tmp_loc)
                cur_loc += cur_degree - tmp_loc
                break
            else:
                warp_len.append(warp_max_nz)
                cur_loc += warp_max_nz
                tmp_loc += warp_max_nz
    pad = np.zeros_like(warp_row)
    warp_4 = np.dstack([warp_row, warp_loc, warp_len, pad]).flatten()
    warp_4.astype(np.int32).tofile(meta_dir + file.stem + '.warp4')
    print(f"{fcnt} {file.stem} finish")



