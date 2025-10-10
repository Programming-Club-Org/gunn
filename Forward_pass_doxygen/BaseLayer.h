// BaseLayer.h
/**
 * @file BaseLayer.h
 * @brief Abstract base class interface for all GNN layers.
 * @details
 * Provides the `forward` method that each Graph Neural Network
 * layer (e.g., GAT, GCN, GraphSAGE) must implement.
 */
#pragma once
#include <vector>
using namespace std;

// BaseLayer provides a standard interface for all GNN layers (GAT, GCN, GraphSAGE, etc.)
class BaseLayer {
public:
    /** Virtual Destructor */
    virtual ~BaseLayer() {}

    // Forward pass interface to be overridden by all derived GNN layers
    /**
     * @brief Executes the forward pass for this layer.
     * @param node_features Input feature matrix of size [num_nodes][feature_dim].
     * @param adjacency_list Adjacency list representing graph connectivity.
     * @return Transformed feature matrix of the same dimensions.
     * @details
     * All derived GNN layers must override this pure virtual method
     * to apply their specific aggregation or attention mechanisms.
     */
    virtual void forward(
        const vector<vector<float>>& node_features,
        vector<vector<float>>& weights
    ) = 0;

    // backward pass takes in gradients with respect to 
    // the previous layer features and computes gradients with respect to
    // weights of current layer and returns the gradients 
    // with respect to features of the current layer.
    virtual vector<vector<float>> backward(
        vector<vector<float>>& grad_prev_features
    ) = 0;
};
