// GraphSage.h

#pragma once
#include "BaseLayer.h"
#include <vector>
using namespace std;

// Implements GraphSage Layer which aggregates the neighbouring features
// using mean aggregation and CONCATENATES them with node's own features
// and then performs linear transformation.
/**
 * @class GraphSAGELayer
 * @brief A GNN layer implementing the GraphSAGE operation.
 * @details
 * Inherits from BaseLayer and defines the forward pass that
 * combines neighbor information with self-information.
 */
class GraphSAGELayer : public BaseLayer {
public:
    // This constructor initialises the Layer with input and output dimensions
    // performs Xavier initialisation of weight matrix.
    /**
     * @brief Constructs a GraphSAGELayer with specified dimensions.
     * @param input_dim   Size of each input node feature vector.
     * @param output_dim  Desired size of each output node feature vector.
     * @details
     * Initializes the internal weight matrix using Xavier initialization
     * with uniform sampling in [−limit, +limit], where  
     * limit = sqrt(6.0f / (input_dim + output_dim)).
     */
    GraphSAGELayer(int input_dim, int output_dim);

    // forward pass computes updated node features by aggregating neighbour features,
    // concatenating with self-features and multiplying with weight matrix followed by ReLU.
    /**
     * @brief Executes the forward pass of the GraphSAGE layer.
     * @param node_features   Input matrix of node features ([num_nodes][input_dim]).
     * @param adjacency_list  Graph adjacency list ([num_nodes][variable neighbors]).
     * @return                Updated feature matrix ([num_nodes][output_dim]).
     * @details
     * For each node:
     *  - Compute the mean of its neighbors’ features  
     *  - Concatenate this aggregated vector with the node’s own features  
     *  - Apply a linear transformation and then ReLU  
     */
    vector<vector<float>> forward(
        const vector<vector<float>>& node_features, // node-feature matrix:[number of nodes][input_dim]
        const vector<vector<int>>& adjacency_list   // represents the graph
    ) override;

private:
    /** Number of features in each input node vector. */
    int input_dim;              // dimension of input features
    /** Number of features in each output node vector. */
    int output_dim;             // dimension of output features
    /** Weight matrix ([2 * input_dim] × [output_dim]) for the linear transform. */
    vector<vector<float>> weight_matrix; // weight matrix of shape [input_dim][output_dim]
    
    // Applies ReLU activation to single float value
    /**
     * @brief Applies the ReLU activation to a scalar.
     * @param x  Input value.
     * @return   std::max(0.0f, x).
     */
    float relu(float x);

    // Aggregates features of the neighbours of this node using mean aggregation.
    /**
     * @brief Aggregates neighbor features by computing their mean.
     * @param node Index of the target node.
     * @param node_features Full feature matrix ([num_nodes][input_dim]).
     * @param adjacency_list  Graph adjacency list.
     * @return Mean‐pooled neighbor feature vector ([input_dim]).
     */
    vector<float> aggregate_neighbors_mean(
        int node,                                   // index of the central node
        const vector<vector<float>>& node_features, // input node feature matrix
        const vector<vector<int>>& adjacency_list   // graph representation
    );

    // CONCATENATES node's own features to the aggregated neighbour features
    // resulting feature vector is of size 2*input_dim.
    /**
     * @brief Concatenates a node’s own features with its neighbors’ features.
     * @param self_features Feature vector of the node ([input_dim]).
     * @param neighbor_features  Aggregated neighbor vector ([input_dim]).
     * @return Concatenated vector ([2 * input_dim]).
     */
    vector<float> concatenate_self_and_neighbors(
        const vector<float>& self_features,     // node's own feature vector
        const vector<float>& neighbor_features  // aggregated neighbour-feature vector
    );

    // Applies weight matrix to the aggregated feature vector to compute
    // output for a single output dimension
    /**
     * @brief Applies a linear transformation to the concatenated features.
     * @param concat_features Concatenated feature vector ([2 * input_dim]).
     * @param output_index Index of the output feature to compute.
     * @return Scalar feature value for that output dimension.
     */
    float linear_transform(
        const vector<float>& concat_features, // the concatenated feature vector
        int output_index                      // index of output dimension being computed
    );
};
