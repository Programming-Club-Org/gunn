// GATL.h
#pragma once
#include "BaseLayer.h"
#include <vector>
using namespace std;

// Implements a layer of Graph Attention Network(GAT)
// It takes into account the importance of each neighbour also in aggregation.
// It uses self-attention mechanism on graphs to compute this importance
/**
 * @class GATLayer
 * @brief Single layer of a Graph Attention Network.
 */
class GATLayer : public BaseLayer {
public:
    int input_dim, output_dim;  // Input and output dimension
    vector<vector<float>> W;    // Weight matrix for linear transformation
    vector<float> a;            // Attention vector used for computing attention coefficients

    // Constructor initializes the GAT layer with input and output dimensions
    // and performs Xavier initialization for weights and attention parameters.
    /**
     * @brief Create a GAT layer.
     * @param input_dim  Size of input features.
     * @param output_dim Size of output features.
     */
    GATLayer(int input_dim, int output_dim);

    // Forward pass computes the updated node features based on attention mechanism.
    // It projects input features, computes attention scores with neighbours, applies softmax,
    // aggregates neighbour features weighted by attention, and applies ReLU.
    /**
     * @brief Run the layer on node features.
     * @param node_features Matrix [num_nodes][input_dim].
     * @param adjacency_list Adjacency list [num_nodes].
     * @return Matrix [num_nodes][output_dim].
     */
    vector<vector<float>> forward(
        const vector<vector<float>>& node_features, // node-feature matrix:[number of nodes][input_dim]
        const vector<vector<int>>& adjacency_list   // represents the graph
    ) override;

private:
    // Applies ReLU activation to a single float value
    /**
     * @brief ReLU activation.
     * @param x Input value.
     * @return  max(0, x).
     */
    float relu(float x);

    // Applies LeakyReLU activation with a configurable alpha slope for negative inputs.
    /**
     * @brief Leaky ReLU activation.
     * @param x  Input value.
     * @param alpha Negative slope (default 0.2).
     * @return x if x>0 else alpha*x.
     */
    float leaky_relu(float x, float alpha = 0.2f);

    // Applies weight matrix to a single node's feature vector to transform feature vector of size output_dim.
    /**
     * @brief Project one node’s features.
     * @param features Vector [input_dim].
     * @return Vector [output_dim].
     */
    vector<float> linear_transform(
        const vector<float>& features // Input feature vector of a node
    );

    // Computes attention score (unnormalised) for node i and node j
    // using attention vector applied to concatenation of projected features of node i and node j
    /**
     * @brief Raw attention score between two nodes.
     * @param z_i Features of node i.
     * @param z_j Features of node j.
     * @return Unnormalized score.
     */
    float compute_attention_score(
        const vector<float>& z_i, // feature vector of node i
        const vector<float>& z_j  // feature vector of node j (neighbour)
    );

    // Applies softmax function to a vector of unnormalised attention scores
    // returns a vector of normalised attention coefficients
    /**
     * @brief Softmax over a list of scores.
     * @param scores Raw scores.
     * @return Normalized scores.
     */
    vector<float> softmax(
        const vector<float>& scores  // Unnormalised attention scores for a node and its neighbours
    );
};
