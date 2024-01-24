#include "spmm_maxk.h"
#include "data.h"
#include <string>
#include <iostream>
#define CONSTINT const int

using namespace std;

extern string base_dir, graph;

const int WARPS_PER_BLOCK = 12;
const int EXT_WARP_DIM = 32;

#define DIM_MUL_N 1
#define DIM_MUL(x) ((x + DIM_MUL_N - 1) / DIM_MUL_N) * DIM_MUL_N

__global__ void spmm_kernel_opt2_sparse_v3(const int *_warp4, const int *idx, const float *val, const float *vin_data, const u_int8_t *vin_selector, float *vout, const int num_v, const int num_e, const int feat_in, const int dim_sparse, const int num_warps)
{
    const int4 *warp4 = reinterpret_cast<const int4 *>(_warp4);
    extern __shared__ float out_cache[];

    const int total_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_warpid = total_tid / EXT_WARP_DIM;
    const int laneid = threadIdx.x % EXT_WARP_DIM;
    const int wid = threadIdx.x / EXT_WARP_DIM;

    const int sparse_wid = wid * (EXT_WARP_DIM / dim_sparse) + laneid / dim_sparse;

    const int sparse_laneid = laneid % dim_sparse;

    int4 sparse_w_info, w_info;
    int sparse_warp_row, sparse_warp_loc, sparse_warp_len;
    int warp_row, warp_loc, warp_len;

    if (total_warpid < num_warps)
    {
        w_info = warp4[total_warpid];
        warp_row = w_info.x;
        warp_loc = w_info.y;
        warp_len = w_info.z;

        if (dim_sparse < 32 && blockIdx.x * blockDim.x / EXT_WARP_DIM + sparse_wid < num_warps)
        {
            sparse_w_info = warp4[blockIdx.x * blockDim.x / EXT_WARP_DIM + sparse_wid];
            sparse_warp_row = sparse_w_info.x;
            sparse_warp_loc = sparse_w_info.y;
            sparse_warp_len = sparse_w_info.z;
        }
    }

#pragma unroll
    for (int ext = 0; ext < (feat_in + EXT_WARP_DIM - 1) / EXT_WARP_DIM; ext++)
    {
        out_cache[threadIdx.x + ext * blockDim.x] = 0;
    }

    if (total_warpid >= num_warps || blockIdx.x * blockDim.x / EXT_WARP_DIM + sparse_wid >= num_warps)
        return;

    __syncthreads();

    if (dim_sparse < 32)
    {
        if (sparse_wid < blockDim.x / EXT_WARP_DIM && laneid / dim_sparse < EXT_WARP_DIM / dim_sparse)
        {
            for (int i = 0; i < sparse_warp_len; i++)
            {

                int nz_loc = sparse_warp_loc + i;

                // use gcn style left value ?
                float left_val = __ldg(val + nz_loc);
                int right_loc = __ldg(idx + nz_loc) * DIM_MUL(dim_sparse) + sparse_laneid;
                float right_val = vin_data[right_loc];

                out_cache[sparse_wid * feat_in + vin_selector[right_loc]] += left_val * right_val;

                //  atomicAdd_block(&out_cache[sparse_wid * feat_in + __ldg(vin_selector + right_loc)], left_val * right_val);
            }
        }
        __syncthreads();
    }
    else
    {
        for (int i = 0; i < warp_len; i++)
        {
            for (int l = laneid; l < dim_sparse; l += 32)
            {
                int nz_loc = warp_loc + i;
                float left_val = __ldg(val + nz_loc);
                int right_loc = __ldg(idx + nz_loc) * DIM_MUL(dim_sparse) + l;
                float right_val = vin_data[right_loc];
                out_cache[wid * feat_in + vin_selector[right_loc]] += left_val * right_val;
                // atomicAdd_block(&out_cache[wid * feat_in + __ldg(vin_selector + right_loc)], left_val * right_val);       
            }
        }
        __syncthreads();
    }


#pragma unroll
    for (int ext = 0; ext < (feat_in + EXT_WARP_DIM - 1) / EXT_WARP_DIM; ext++)
    {
        atomicAdd(&vout[warp_row * feat_in + laneid + ext * EXT_WARP_DIM], out_cache[wid * feat_in + laneid + ext * EXT_WARP_DIM]);
    }
}

void SPMM_MAXK::run(int dim)
{
    int shared_size = WARPS_PER_BLOCK * dim * sizeof(float);

    spmm_kernel_opt2_sparse_v3<<<grid, block, shared_size>>>(_warp4, idx, val, vin, vin_sparse_selector, vout, num_v, num_e, dim, dim_sparse, num_warps);
}

double SPMM_MAXK::do_test(bool timing, int dim)
{
    this->num_warps = cuda_read_array(&this->_warp4, "../w" + to_string(WARPS_PER_BLOCK) + "_nz" + "64_warp_4/" + this->_graph + ".warp4") / 4;
    int block_num = (num_warps + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    if (!timing)
    {
        cout << "block num = " << block_num << endl;
    }

    grid.x = block_num;
    block.x = WARPS_PER_BLOCK * EXT_WARP_DIM;

    double ret = timing_body(timing, dim);

    cudaFree(this->_warp4);
    return ret;
}