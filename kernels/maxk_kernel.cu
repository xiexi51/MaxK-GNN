#include <iostream>
#include <random>
#include <chrono>

#define timestamp(__var__) auto __var__ = std::chrono::system_clock::now();
inline double getDuration(std::chrono::time_point<std::chrono::system_clock> a,
                          std::chrono::time_point<std::chrono::system_clock> b)
{
    return std::chrono::duration<double>(b - a).count();
}

const int WARPS_PER_BLOCK = 16;
const int N = 232965 / 256 * 256;
// const int N = 2449029 / 256 * 256;
// const int N = 89250 / 256 * 256;
// const int N = 132534 / 256 * 256;
// const int N = 716847 / 256 * 256;

const int dim_origin = 256, dim_k = 32;

using namespace std;

__global__ void topk(uint8_t *data, uint8_t *value, uint8_t *index, uint k)
{
    // __shared__ uint8_t cache[256*16];
    extern __shared__ uint8_t cache[];

    // const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int wid = threadIdx.x / 32;
    const int laneid = threadIdx.x % 32;

    *((int2*)cache + threadIdx.x) = *((int2*)data + blockIdx.x * blockDim.x + threadIdx.x);

    __syncwarp();
    uint8_t low = 0, high = 255, mid = 127;
    uint count;
    #pragma unroll
    for(int i = 0; i < 8; i++){
        count = 0;
        #pragma unroll
        for(int j = 0; j < 8; j++){
            count += cache[laneid * 8 + j] > mid;
        }
        count += __shfl_down_sync(0xffffffff, count, 16);
        count += __shfl_down_sync(0xffffffff, count, 8);
        count += __shfl_down_sync(0xffffffff, count, 4);
        count += __shfl_down_sync(0xffffffff, count, 2);
        count += __shfl_down_sync(0xffffffff, count, 1);
        if(count < k){
            high = mid;
        }
        // else if(count > k){
        //     low = mid;
        // }
        // else{
        //     break;
        // }
        else{
            low = mid;
        }
        mid = (low + high) / 2;
        
    }
    cache[WARPS_PER_BLOCK * (dim_origin + 2 * k) + wid] = 0;
    __syncthreads();

    // int total_loc = 0;
    #pragma unroll
    for(int ext = 0; ext < 8; ext++){
        uint8_t total_loc = cache[WARPS_PER_BLOCK * (dim_origin + 2 * k) + wid];
        if(total_loc >= k){
            break;
        }
        uint8_t val = cache[wid * dim_origin + laneid + ext * 32];
        bool choose = val > mid;
        // unsigned mask = __ballot_sync(__activemask(), choose);
        unsigned mask = __ballot_sync(0xffffffff, choose);

        int loc = __popc(mask & ((1 << laneid) - 1));
        if(choose && total_loc + loc < k){
            cache[WARPS_PER_BLOCK * dim_origin + wid * k + total_loc + loc] = val;
            cache[WARPS_PER_BLOCK * (dim_origin + k) + wid * k + total_loc + loc] = laneid + ext * 32;
        }
        if(laneid == 31){
            cache[WARPS_PER_BLOCK * (dim_origin + 2 * k) + wid] += loc;
        }
        __syncwarp();
    }

    __syncthreads();
    if(wid < WARPS_PER_BLOCK / 4){
        *((int*)value + blockIdx.x * blockDim.x / 4 + threadIdx.x) = *((int*)cache + WARPS_PER_BLOCK * dim_origin / 4 + threadIdx.x);
        *((int*)index + blockIdx.x * blockDim.x / 4 + threadIdx.x) = *((int*)cache + WARPS_PER_BLOCK * (dim_origin + k) / 4 + threadIdx.x);
    }
        
}

int main() {
    cout<<"N = "<< N << ", dim_origin = " << dim_origin << ", dim_k = " << dim_k << ", preparing data..."<<endl;

    uint8_t *data, *value, *index;

    // Allocate unified memory
    cudaMallocManaged(&data, N * dim_origin * sizeof(uint8_t));
    cudaMallocManaged(&value, N * dim_k * sizeof(uint8_t));
    cudaMallocManaged(&index, N * dim_k * sizeof(uint8_t));

    // Initialize data with random values
    std::default_random_engine engine;
    engine.seed(123);
    std::uniform_int_distribution<uint8_t> distribution(0, 255);
    for (int i = 0; i < N * dim_origin; ++i) {
        data[i] = distribution(engine);
    }
    // for (int i = 0; i < N * dim_k; i++){
    //     value[i] = 0;
    //     index[i] = 0;
    // }

    cout << "data ready, testing..." << endl;


    int shared_size = WARPS_PER_BLOCK * (dim_origin + 2 * dim_k + 1);

    int times = 10;
    // warmup
    for (int i = 0; i < times; i++)
    {
        topk<<<N / WARPS_PER_BLOCK, WARPS_PER_BLOCK * 32, shared_size>>>(data, value, index, dim_k);
    }
    cudaDeviceSynchronize();
    double measured_time = 0;
    for (int i = 0; i < times; i++)
    {
        timestamp(t0);
        topk<<<N / WARPS_PER_BLOCK, WARPS_PER_BLOCK * 32, shared_size>>>(data, value, index, dim_k);
        cudaDeviceSynchronize();
        timestamp(t1);
        measured_time += getDuration(t0, t1);
    }

    cout << "maxk kernel time = " << measured_time / times * 1000 << " ms" <<endl;
    
    // topk<<<N / WARPS_PER_BLOCK, WARPS_PER_BLOCK * 32, shared_size>>>(data, value, index, dim_k);

    // Wait for GPU to finish before accessing on host

    // cout<<"finish"<<endl;

    // for(int i = 0; i < N; i++){
    //     for(int j = 0; j < dim_k; j++)
    //         cout << (int)value[i * dim_k + j] << " ";
    //     cout<<endl;
    // }
    // cout<<endl<<endl;

    // for(int i = 0; i < N; i++){
    //     for(int j = 0; j < dim_k; j++)
    //         cout << (int)index[i * dim_k + j] << " ";
    //     cout<<endl;
    // }
    // cout<<endl<<endl;
    


    // Free unified memory
    cudaFree(data);
    cudaFree(value);
    cudaFree(index);

    return 0;
}
