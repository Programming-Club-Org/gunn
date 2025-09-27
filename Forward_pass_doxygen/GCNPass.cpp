
#include "GCNPass.h"
#include <iostream>
using namespace std;

vector_Layer::vector_Layer(int hidden_layers, int input_dim, int output_dim, int hidden_dim) {
        network_layer.resize(hidden_layers + 2);
        for (int i = 0; i < hidden_layers + 2; ++i) {
        int in_dim = (i == 0) ? input_dim : hidden_dim;
        int out_dim = (i == hidden_layers+1) ? output_dim : hidden_dim;
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
        vector<int> degrees(n_nodes);
        for (int i = 0; i < n_nodes; i++) {degrees[i] = g.adjacency_list[i].size();}

        vector<vector<float>> curr_features = g.node_features;

        for (int layer = 0; layer < hidden_layers + 2; layer++) {
            vector<vector<float>> aggregated(n_nodes, vector<float>((layer == hidden_layers + 1) ? out_dim : hidden_dim, 0.0f));
            for (int i = 0; i < n_nodes; i++) {
                aggregated[i] = network_layer[layer].aggregate_neighbors(i, curr_features, g.adjacency_list, degrees);
            }
            network_layer[layer].forward(aggregated, g.adjacency_list);
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
        int n=grad_prev.size();
        int m=grad_prev[0].size();
        cout << "Grad of last layer\n";
        for(int i=0;i<n;i++) {
            for(int j=0;j<m;j++) {
                cout << grad_prev[i][j] << " ";
            }cout << "\n";
        }
        for (int i = num_layers - 2; i >= 0; i--) {
            grad_prev = network_layer[i].backward(grad_prev);
        }

        for (int i = 0; i < num_layers; i++) {
            int n = network_layer[i].weight_matrix.size();
            int m = network_layer[i].weight_matrix[0].size();
            cout << "Grad weight matrix\n";
            for(int j = 0; j < n; j++) {
                for(int k = 0; k < m; k++) {
                    cout << network_layer[i].grad_weight_matrix[j][k] << " ";
                }cout << "\n";
            }
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < m; k++) {
                    network_layer[i].weight_matrix[j][k] -= learning_rate * network_layer[i].grad_weight_matrix[j][k];
                    //cout << network_layer[i].weight_matrix[j][k] << " ";
                }
            }
        }
    }