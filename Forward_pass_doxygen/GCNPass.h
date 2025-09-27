#pragma once
#include <vector>
#include "GCNTest.h"
#include "Graph.h"
#include "GraphReader.h"
#include "BaseLoss.h"
using namespace std;

class vector_Layer {

public:
    vector<GCNTestLayer> network_layer;

    vector_Layer(int hidden_layers, int input_dim, int output_dim, int hidden_dim);
    
    void forward_pass (
        int out_dim,
        int hidden_dim,
        int hidden_layers,
        const Graph& g
    );

    void backward_pass
    (
        float learning_rate,
        vector<vector<float>> expected_value
    );
};