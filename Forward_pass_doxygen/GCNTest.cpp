// GCNL.cpp

#include "GCNTest.h"
#include <random>
#include <algorithm>
#include <cmath>

// Xavier Initialization
GCNTestLayer::GCNTestLayer(int input_dim, int output_dim) : input_dim(input_dim), output_dim(output_dim) {
    weight_matrix.resize(input_dim, vector<float>(output_dim,1.0f));
}

// ReLU activation
float GCNTestLayer::relu(float x) {
    return max(0.0f, x);
}

// Aggregates normalized neighbor features for a node
vector<float> GCNTestLayer::aggregate_neighbors(
    int node,
    const vector<vector<float>>& node_features,
    const vector<vector<int>>& adjacency_list,
    const vector<int>& degrees
) {
    vector<float> aggregated(input_dim, 0.0f);
    for (int neighbor : adjacency_list[node]) {
        float normalization = sqrt(degrees[node] * degrees[neighbor]);
        if (normalization != 0.0f) {
            for (int d = 0; d < input_dim; d++) {
                aggregated[d] += node_features[neighbor][d] / normalization;
            }
        }
    }
    return aggregated;
}

// Applies weight matrix for a given output dimension
float GCNTestLayer::linear_transform(
    const vector<float>& aggregated_features,
    int output_index
) {
    float val = 0.0f;
    for (int d = 0; d < input_dim; d++) {
        val += aggregated_features[d] * weight_matrix[d][output_index];
    }
    return val;
}

// Forward pass for GCN Layer
void GCNTestLayer::forward(
    const vector<vector<float>>& node_features,
    const vector<vector<int>>& adjacency_list
) {
    int n_nodes = node_features.size();
    vector<vector<float>> updated_features(n_nodes, vector<float>(output_dim, 0.0f));

    // Precompute degrees
    vector<int> degrees(n_nodes);
    for (int i = 0; i < n_nodes; i++) {
        degrees[i] = adjacency_list[i].size();
    }

    for (int i = 0; i < n_nodes; i++) {
        vector<float> aggregated = aggregate_neighbors(i, node_features, adjacency_list, degrees);
        for (int o = 0; o < output_dim; o++) {
            float val = linear_transform(aggregated, o);
            updated_features[i][o] = relu(val);
        }
    }

    layer_features=updated_features;
}

// Backward pass for GCN Layer
vector<vector<float>>  GCNTestLayer::backward(
        vector<vector<float>>& grad_prev_features
) {
    vector<vector<float>> grad_curr_features;
    int n=grad_prev_features.size();
    int num_layer_features=grad_prev_features[0].size();
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < num_layer_features; j++) {
            grad_weight_matrix[i][j]=0;
            for(int row = 0; row < n; row++) {
                grad_weight_matrix[i][j]+=grad_prev_features[row][j]*layer_features[row][i];
            }
            grad_curr_features[i][j]=0;
            for(int row = 0; row < n; row++) {
                grad_curr_features[i][j]+=grad_prev_features[i][row]*weight_matrix[j][row];
            }
        }
    }
    return grad_curr_features;
}
