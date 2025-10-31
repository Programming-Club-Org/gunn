#include "Forward_kernel.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;
__global__ void dkernel_forward(
    float** d_inp_features, 
    float** d_weights, 
    float** d_out_features,
    int* d_row_ptr,
    int* d_col_index,
    int num_nodes, 
    int num_in_features, 
    int num_out_features
) {
    int node_id=blockIdx.x*blockDim.x+threadIdx.x;
    if(node_id>=num_nodes) {return;}
    
    int row_start=d_row_ptr[node_id];
    int row_end=d_row_ptr[node_id+1];
    int deg_i=row_end-row_start;
    if (deg_i == 0) {
        for (int o = 0; o < num_out_features; ++o) {d_out_features[node_id][o]=0.0f;}
        return;
    }

    float* aggregated=new float[num_in_features];
    for(int i = 0; i < num_in_features; i++) {
        aggregated[i]=0.0f;
    }

    for (int k = row_start; k < row_end; k++) {
        int neighbor_id=d_col_index[k];
        float* neighbor_features=d_inp_features[neighbor_id];

        int deg_j=d_row_ptr[neighbor_id+1]-d_row_ptr[neighbor_id]; 
        float norm=sqrtf((float)(deg_i*deg_j));

        for (int f = 0; f < num_in_features; f++) {
            aggregated[f]+=(neighbor_features[f])/norm;
        }
    }

    for (int o = 0; o < num_out_features; ++o) {
        float sum=0.0f;
        for (int f = 0; f < num_in_features; ++f) {
            sum+=aggregated[f]*d_weights[f][o];
        }
        d_out_features[node_id][o]=sum;
    }
    delete[] aggregated;
}

vector<vector<float>> forward_kernel_call(
    int num_nodes,
    int num_in_features,
    int num_out_features,
    const vector<vector<float>>& input_features,
    vector<vector<float>>& weight_matrix,
    const vector<vector<int>>& adjacency_list
) {
    vector<vector<float>> out_features(num_nodes,vector<float>(num_out_features));

    vector<int> row_ptr;
    vector<int> col_index;
    row_ptr.push_back(0);
    for(int i = 0; i < num_nodes; i++) {
        for(int neighbor : adjacency_list[i]) {
            col_index.push_back(neighbor);
        }
        row_ptr.push_back(col_index.size());
    }
    int row_ptr_size=(int)row_ptr.size();
    int col_index_size=(int)col_index.size(); 

    int* d_row_ptr;
    int* d_col_index;
    cudaMalloc((void**)&d_row_ptr,row_ptr_size*sizeof(int));
    cudaMalloc((void**)&d_col_index,col_index_size*sizeof(int));
    cudaMemcpy(d_row_ptr,row_ptr.data(),row_ptr_size*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_index,col_index.data(),col_index_size*sizeof(int),cudaMemcpyHostToDevice);

    float** d_inp_features;
    cudaMalloc(&d_inp_features,num_nodes*sizeof(float*));
    for (int i = 0; i < num_nodes; ++i) {
        float* d_node_feature;
        cudaMalloc(&d_node_feature,num_in_features*sizeof(float));
        cudaMemcpy(d_node_feature,input_features[i].data(),num_in_features*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_inp_features+i,&d_node_feature,sizeof(float*), cudaMemcpyHostToDevice);
    }

    float** d_weights;
    cudaMalloc(&d_weights,num_in_features*sizeof(float*));
    for (int i = 0; i < num_in_features; ++i) {
        float* d_weight;
        cudaMalloc(&d_weight, num_out_features * sizeof(float));
        cudaMemcpy(d_weight, weight_matrix[i].data(),
                num_out_features * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_weights + i, &d_weight, sizeof(float*), cudaMemcpyHostToDevice);
    }


    float** d_out_features;
    cudaMalloc(&d_out_features,num_nodes*sizeof(float*));
    for (int i = 0; i < num_nodes; ++i) {
        float* d_node_out;
        cudaMalloc(&d_node_out, num_out_features * sizeof(float));
        cudaMemcpy(d_out_features + i, &d_node_out, sizeof(float*), cudaMemcpyHostToDevice);
    }

    int threads_per_block = 256;
    int blocks = (num_nodes + threads_per_block - 1) / threads_per_block;

    dkernel_forward<<<blocks, threads_per_block>>>(
        d_inp_features,
        d_weights,
        d_out_features,
        d_row_ptr,
        d_col_index,
        num_nodes,
        num_in_features,
        num_out_features
    );
    cudaDeviceSynchronize();

    for (int i = 0; i < num_nodes; ++i) {
        float* d_node_out=nullptr;
        cudaMemcpy(&d_node_out, d_out_features + i, sizeof(float*), cudaMemcpyDeviceToHost);
        cudaMemcpy(out_features[i].data(), d_node_out,
                num_out_features*sizeof(float), cudaMemcpyDeviceToHost);
    }

    return out_features;
}