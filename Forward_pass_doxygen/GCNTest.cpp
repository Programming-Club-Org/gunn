// GCNL.cpp

#include "GCNTest.h"
#include <random>
#include <algorithm>
#include <cmath>
#include <iostream>

GCNTestLayer::GCNTestLayer() : input_dim(0), output_dim(0) {}

GCNTestLayer::GCNTestLayer(int input_dim, int output_dim) : input_dim(input_dim), output_dim(output_dim) {
    weight_matrix.resize(input_dim, vector<float>(output_dim,0.0f));
    grad_weight_matrix.resize(input_dim, vector<float>(output_dim, 0.0f));
}

// ReLU activation
float GCNTestLayer::relu(float x) {
    return max(0.0f,x);
}

// Aggregates normalized neighbor features for a node
vector<float> GCNTestLayer::aggregate_neighbors(
    int node,
    const vector<vector<float>>& node_features,
    const vector<vector<int>>& adjacency_list,
    const vector<int>& degrees
) {
    int s=node_features[0].size();
    vector<float> aggregated(s, 0.0f);
    for (int neighbor : adjacency_list[node]) {
        float normalization = sqrt(degrees[node] * degrees[neighbor]);
        if (normalization != 0.0f) {
            for (int d = 0; d < s; d++) {
                aggregated[d] += node_features[neighbor][d] / normalization;
            }
        }
    }
    return aggregated;
}

// Applies weight matrix for a given output dimension
float GCNTestLayer::linear_transform(
    const vector<float>& aggregated_features,
    vector<vector<float>>& weights,
    int output_index
) {
    int n=weights.size();
    int m=weights[0].size();
    float val = 0.0f;
    for (int d = 0; d < n; d++) {
        val += aggregated_features[d] * weights[d][output_index];
    }
    return val;
}

// Forward pass for GCN Layer
void GCNTestLayer::forward(
    const vector<vector<float>>& node_features,
    vector<vector<float>>& weights
) {
    int n_nodes = node_features.size();
    vector<vector<float>> updated_features(n_nodes, vector<float>(input_dim, 0.0f));

    cached_input_features=node_features;
    cached_linear_output.assign(n_nodes,vector<float>(input_dim,0.0f));

    // Precompute degrees
    for (int i = 0; i < n_nodes; i++) {
        //vector<float> aggregated = aggregate_neighbors(i, node_features, adjacency_list, degrees);
        for (int o = 0; o < input_dim; o++) {
            float val = linear_transform(node_features[i], weights, o);
            cached_linear_output[i][o] = val;
            updated_features[i][o] = relu(val);
        }
    }

    layer_features=updated_features;
}

// Backward pass for GCN Layer
vector<vector<float>>  GCNTestLayer::backward(
        vector<vector<float>>& grad_prev_features
) {
    int n=grad_prev_features.size();

    vector<vector<float>> grad_after_relu = grad_prev_features;

    cout << "Gradient of previous features\n";
    for(int i=0;i<n;i++) {
        for(int j=0;j<output_dim;j++) {
            cout << grad_prev_features[i][j] << " ";
        }cout << "\n";
    }cout << "\n";
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < input_dim; j++) {
            //cout << cached_linear_output[i][j] << " ";
            if (cached_linear_output[i][j]<=0.0f) {
                grad_after_relu[i][j]=0.0f;
            }
        }
    }

    for (auto &row : grad_weight_matrix) {
        fill(row.begin(), row.end(), 0.0f);
    }

    for(int i = 0; i < input_dim; i++) {
        for(int j = 0; j < output_dim; j++) {
            grad_weight_matrix[i][j]=0.0f;
            for(int row = 0; row < input_dim; row++) {
                //cout << i << " " << j << " " << row << "\n";
                //cout << grad_after_relu[row][j] << " " << cached_linear_output[row][i] << "\n";
                grad_weight_matrix[i][j]+=grad_after_relu[row][j]*cached_linear_output[row][i];
            }
        }
    }
    cout << "Grad weight matrix\n";
            for(int j = 0; j < input_dim; j++) {
                for(int k = 0; k < output_dim; k++) {
                    cout << grad_weight_matrix[j][k] << " ";
                }cout << "\n";
            }cout << "\n"; 

    vector<vector<float>> grad_curr_features(n, vector<float>(input_dim,0.0f));
    for(int i=0;i<input_dim;i++) {
        for(int j=0;j<output_dim;j++) {
            cout << weight_matrix[i][j] << " ";
        }cout << "\n";
    }cout << "\n";
    for(int node = 0; node < n; node++) {
        for(int j = 0; j < input_dim; j++) {
            grad_curr_features[node][j] = 0.0f;
            cout << "Testing\n";
            for(int k = 0; k < output_dim; k++) {
                cout << grad_prev_features[node][k] << " " << weight_matrix[j][k] << " ";
                grad_curr_features[node][j] += grad_prev_features[node][k] * weight_matrix[j][k];
                cout << grad_curr_features[node][j] << "\n";
            }
        }
    }
    cout << "Grad curr features\n";
            for(int j = 0; j < n; j++) {
                for(int k = 0; k < input_dim; k++) {
                    cout << grad_curr_features[j][k] << " ";
                }cout << "\n";
            }cout << "\n";  
    return grad_curr_features;
}
