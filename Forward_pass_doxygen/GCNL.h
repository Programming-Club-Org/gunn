// GCNL.h

#pragma once
#include "BaseLayer.h"
#include <vector>
using namespace std;

// GCNLayer implements Graph Convolution Neural Network
// performs feature aggregation ONLY from neighbours and then does linear transformation.
/**
 * @class GCNLayer
 * @brief Implements the core Graph Convolution operation.
 * @details
 * The GCNLayer:
 *  - Computes normalization based on node degrees  
 *  - Aggregates neighbor feature vectors  
 *  - Transforms the aggregated vector via a weight matrix  
 *  - Applies ReLU activation to each output dimension  
 */
class GCNLayer : public BaseLayer {
public:
    // This constructor initialises the Layer with input and output dimensions
    // performs Xavier initialisation of weight matrix.
    /**
     * @brief Constructs a GCNLayer with given dimensions.
     * @param input_dim  Dimensionality of input feature vectors.
     * @param output_dim Dimensionality of output feature vectors.
     * @details
     * The weight matrix is initialized using Xavier initialization,
     * sampling uniformly from [−limit, +limit], where  
     * limit = sqrt(6 / (input_dim + output_dim)).
     */
    GCNLayer(int input_dim, int output_dim);

    // forward pass computes the updated node features 
    // using the input features and the adjacency list
    /**
     * @brief Executes a forward pass of the GCN layer.
     * @param node_features Input feature matrix of shape [num_nodes][input_dim].
     * @param adjacency_list Graph adjacency list representation.
     * @return Updated feature matrix [num_nodes][output_dim].
     * @details
     * -Precompute node degrees for normalization  
     * -Aggregate neighbor features with normalization  
     * -Apply linear transformation and ReLU per node  
     */
    vector<vector<float>> forward(
        const vector<vector<float>>& node_features, // feature matrix-[number of nodes][input_dim]
        const vector<vector<int>>& adjacency_list // represents graph structure
    ) override;

private:
    int input_dim;              /** Dimension of input features*/
    int output_dim;             /** Dimension of output features*/
    vector<vector<float>> weight_matrix; /** Weight matrix of shape [input_dim][output_dim]*/
    
    /**
     * @brief Applies the ReLU activation to a scalar.
     * @param x Input value.
     * @return  max(0, x).
     */
    float relu(float x); // Applies ReLU function to a single value (Activation function)

    // Aggregates normalised neighbour features for a given node.
    // Each neighbour's features are first scaled down by the inverse of
    // the product of degrees of node and the neighbour, ensuring normalisation.
    /**
     * @brief Aggregates normalized neighbor features for a single node.
     * @param node Index of the target node.
     * @param node_features Full feature matrix [num_nodes][input_dim].
     * @param adjacency_list Graph adjacency list.
     * @param degrees Precomputed degree of each node.
     * @return Aggregated feature vector of length input_dim.
     */
    vector<float> aggregate_neighbors(
        int node,                                   // the centre node
        const vector<vector<float>>& node_features, // Feature matrix of nodes
        const vector<vector<int>>& adjacency_list,  // graph adjacency list
        const vector<int>& degrees                  // pre-computed degrees of each node
    );

    // Applies weight matrix to the aggregated neighbour features to compute
    // the output for a single output dimension
    /**
     * @brief Computes the weighted sum for one output dimension.
     * @param aggregated_features Aggregated neighbor feature vector.
     * @param output_index Which output dimension to compute.
     * @return Scalar feature value for that output dimension.
     */
    float linear_transform(
        const vector<float>& aggregated_features, // Aggregated and normalised neighbour features
        int output_index                          // index of output dimension being computed
    );
};
