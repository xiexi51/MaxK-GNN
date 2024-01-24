#include <iostream>
#include "data.h"
#include "spmm_maxk.h"
#include "spmm_maxk_backward.h"
#include "spmm_cusparse.h"
#include <random>
#include <algorithm>
#include <filesystem>

string base_dir = "../graphs/";

int total_file_cnt, current_file_cnt;

using namespace std;

#define DIM_MUL_N 1
#define DIM_MUL(x) ((x + DIM_MUL_N - 1) / DIM_MUL_N) * DIM_MUL_N

double check_err(float *out, float *out_ref, int len, bool &has_err)
{
    double err_sum = 0;
    bool show = 1;

    has_err = 0;

    for (int i = 0; i < len; i++)
    {
        double err = abs(out[i] - out_ref[i]);
        err_sum += err;
        
        // if (err > 0.1 && has_err == 0)
        if (err > 0.1)
        {
            has_err = 1;
            // cout << "err at " << i << " err = " << err << " ref = " << out_ref[i] <<endl;
        }
    }
    cout << "err sum = " << err_sum << "  ";
    if (err_sum / len < 0.001)
    {
        cout << "validation pass!" << endl;
    }
    else
    {
        cout << "validation fail!" << endl;
    }
    return err_sum;
}

void test_graph(string graph)
{
    int dim_origin = 256;
    int dim_k_list[] = {16, 32, 64, 96, 128, 192};
    int dim_k_limit = 64;

    int *cu_indptr, *cu_indices;
    int v_num = cuda_read_array(&cu_indptr, base_dir + graph + ".indptr") - 1;
    int e_num = cuda_read_array(&cu_indices, base_dir + graph + ".indices");

    float *cu_val;
    cudaMallocManaged(&cu_val, e_num * sizeof(float));

    float *cu_vout_ref;
    float *cu_vin_sparse, *cu_vin_sparse_data, *cu_vout_maxk, *cu_vout_maxk_backward;
    u_int8_t *cu_vin_sparse_selector;
    cudaMallocManaged(&cu_vout_ref, v_num * dim_origin * sizeof(float));
    cudaMallocManaged(&cu_vin_sparse, v_num * dim_origin * sizeof(float));
    cudaMallocManaged(&cu_vin_sparse_data, v_num * DIM_MUL(dim_k_limit) * sizeof(float));
    cudaMallocManaged(&cu_vout_maxk_backward, v_num * DIM_MUL(dim_k_limit) * sizeof(float));
    cudaMallocManaged(&cu_vin_sparse_selector, v_num * DIM_MUL(dim_k_limit) * sizeof(u_int8_t));
    cudaMallocManaged(&cu_vout_maxk, v_num * dim_origin * sizeof(float));
    

    default_random_engine engine;
    engine.seed(123);

    uniform_real_distribution<float> rd(0, 1);

    int input_mode = 1;
    switch (input_mode)
    {
    case 1:
        generate(cu_val, cu_val + e_num, [&]()
                 { return rd(engine); });
        break;
    case 2:
        for (int i = 0; i < e_num; i++)
        {
            cu_val[i] = 1;
        }
        break;

    default:
        break;
    }
    generate(cu_vin_sparse_data, cu_vin_sparse_data + v_num * dim_k_limit, [&]() { return rd(engine); });
    generate(cu_vin_sparse, cu_vin_sparse + v_num * dim_origin, [&]() { return rd(engine); });

    vector<int> sequence(dim_origin);
    iota(sequence.begin(), sequence.end(), 0); 

    SPMM_MAXK maxk(graph, cu_indptr, cu_indices, cu_val, cu_vin_sparse_data, cu_vout_maxk, v_num, e_num, dim_origin);
    SPMM_MAXK_BACKWARD maxk_backward(graph, cu_indptr, cu_indices, cu_val, cu_vin_sparse, cu_vout_maxk_backward, v_num, e_num, dim_origin);
    
    bool check = false;
    bool timing = true;
    double t_cusparse;

    cout << "num graph dim_origin dim_k kernel time(ms)" << endl;

    for (int n = 0; n < sizeof(dim_k_list) / sizeof(int); n++)
    {
        int dim_k = dim_k_list[n];
        if(dim_k > dim_k_limit){
            break;
        }

        string outstr = to_string(current_file_cnt) + "/" + to_string(total_file_cnt) + " " + graph + " " + to_string(dim_origin) + " " + to_string(dim_k);

        vector<int> sample(dim_k);

        for (int i = 0; i < v_num; ++i)
        {
            std::sample(sequence.begin(), sequence.end(), sample.begin(), dim_k, engine);

            for (int j = 0; j < dim_k; ++j)
            {
                float v = rd(engine);
                // float v = cnt++ * 0.01;
                cu_vin_sparse_data[i * DIM_MUL(dim_k) + j] = v;
                cu_vin_sparse_selector[i * DIM_MUL(dim_k) + j] = sample[j];
            }
        }

        for (int i = 0; i < v_num; ++i)
        {
            for (int j = 0; j < dim_origin; ++j)
            {
                cu_vin_sparse[i * dim_origin + j] = 0.0;
            }
            for (int j = 0; j < dim_k; ++j)
            {
                int col = cu_vin_sparse_selector[i * DIM_MUL(dim_k) + j];
                cu_vin_sparse[i * dim_origin + col] = cu_vin_sparse_data[i * DIM_MUL(dim_k) + j];
            }
        }

        maxk.vin_sparse_selector = cu_vin_sparse_selector;
        maxk.dim_sparse = dim_k;

        maxk_backward.vin_sparse_selector = cu_vin_sparse_selector;
        maxk_backward.dim_sparse = dim_k;

        // if(n == 0){
        //     cout << outstr << endl;
        //     spmm_cusparse(cu_indptr, cu_indices, cu_val, cu_vin_sparse, cu_vout_ref, v_num, e_num, dim_origin, 0);
        //     maxk.do_test(false, dim_origin);
        //     bool has_err = 0;
        //     check_err(cu_vout_maxk, cu_vout_ref, v_num * dim_origin, has_err);
        //     break;
        // }

        if(n == 0){
            t_cusparse = spmm_cusparse(cu_indptr, cu_indices, cu_val, cu_vin_sparse, cu_vout_ref, v_num, e_num, dim_origin, 10);
            cout << outstr << " cusparse " << t_cusparse * 1000 << endl;
        }

        double t_maxk = maxk.do_test(true, dim_origin);
        cout << outstr << " maxk " << t_maxk * 1000 << endl;

        double t_maxk_backward = maxk_backward.do_test(true, dim_origin);
        cout << outstr << " maxk_backward " << t_maxk_backward * 1000 << endl;

    }

    cudaFree(cu_indptr);
    cudaFree(cu_indices);
    cudaFree(cu_val);
    cudaFree(cu_vout_ref);
    cudaFree(cu_vin_sparse);
    cudaFree(cu_vin_sparse_data);
    cudaFree(cu_vin_sparse_selector);
    cudaFree(cu_vout_maxk);
    cudaFree(cu_vout_maxk_backward);
}

int main(int argc, char *argv[])
{
    if (argc > 1)
    {
        string arg_graph(argv[1]);
        test_graph(arg_graph);
    }
    else
    {
        string folder_path = base_dir;
        string extension = ".indptr";

        total_file_cnt = 0;
        for (const auto &file : filesystem::directory_iterator(folder_path))
        {
            if (file.path().extension() == extension)
            {
                total_file_cnt++;
            }
        }

        current_file_cnt = 0;
        for (const auto &file : filesystem::directory_iterator(folder_path))
        {
            if (file.path().extension() == extension)
            {
                current_file_cnt++;
                string graph = file.path().stem().string();
                test_graph(graph);
                cudaDeviceSynchronize();
            }
        }
    }

    return 0;
}