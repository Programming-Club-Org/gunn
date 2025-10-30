
#include "GCNPass.h"
#include <iostream>
using namespace std;

vector_Layer::vector_Layer(int hidden_layers, int input_dim, int output_dim, int hidden_dim) {
        network_layer.resize(hidden_layers + 2);
        for (int i = 0; i < hidden_layers + 2; ++i) {
            int in_dim;
            if(i==0) {in_dim=input_dim;}
            else if(i<=hidden_layers) {in_dim=hidden_dim;}
            else if(i==hidden_layers+1) {in_dim=output_dim;}
            int out_dim = (i == hidden_layers) ? output_dim : hidden_dim;
            network_layer[i] = GCNTestLayer(in_dim, out_dim); 
        }
}

void vector_Layer::forward_pass(
        int out_dim,
        int hidden_dim,
        int hidden_layers,
        const Graph& g
    ) {
        int n_nodes = g.num_nodes;
        network_layer[0].layer_features=g.node_features;
        network_layer[0].cached_linear_output=g.node_features;
        vector<vector<float>> curr_features=g.node_features;

        for(int layer = 1; layer < hidden_layers + 2; layer++) {
            network_layer[layer].forward(curr_features,network_layer[layer-1].weight_matrix,g.adjacency_list);
            curr_features = network_layer[layer].layer_features;
        }
    }

void vector_Layer::backward_pass(
        float learning_rate,
        vector<vector<float>> expected_value
    ) {
        int num_layers = network_layer.size();
        BaseLoss CEL;
        vector<vector<float>> grad_prev = CEL.CrossEntropyLoss(network_layer[num_layers - 1].layer_features, expected_value);
        
        for (int i = num_layers - 2; i >= 0; i--) {
            grad_prev = network_layer[i].backward(grad_prev);
        }

        for (int i = 0; i < num_layers; i++) {
            int n = network_layer[i].weight_matrix.size();
            int m = network_layer[i].weight_matrix[0].size();
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < m; k++) {
                    network_layer[i].weight_matrix[j][k] -= learning_rate * network_layer[i].grad_weight_matrix[j][k];
                    //cout << network_layer[i].weight_matrix[j][k] << " ";
                }
            }
        }
    }