#ifndef TEST_KERNEL_CUH
#define TEST_KERNEL_CUH
#include <vector>

__global__ void dkernel_forward(
    float** d_inp_features, 
    float** d_weights, 
    float** d_out_features,
    int* d_row_ptr,
    int* d_col_index,
    int num_nodes, 
    int num_in_features, 
    int num_out_features
);

vector<vector<float>> forward_kernel_call(
    int num_nodes,
    int num_in_features,
    int num_out_features,
    const vector<vector<float>>& input_features,
    vector<vector<float>>& weight_matrix,
    const vector<vector<int>>& adjacency_list
);