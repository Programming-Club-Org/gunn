// main.cpp

#include "Graph.h"              // your Graph class
#include "GraphReader.h"        // read_graph_from_file(...)
#include "GCNTest.h"            // your existing GCNLayer
#include "GCNPass.h"              
#include "output.h"             // OutputConverter API
#include <iostream>
#include <vector>
#include <numeric>
#include <functional>           // for function

int main(int argc, char** argv) {
    int out_dim;
    int hidden_layers;
    int epochs=2;
    int learning_rate=0.05f;
    // 1) Grab the input filename
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <graph_input_file>\n";
        return 1;
    }
    string filename = argv[1];

    // 2) Read the graph
    Graph g = read_graph_from_file(filename);

    // 3) Run one GCN layer
    cout << "Enter output feature dimension: ";
    cin >> out_dim;
    cout << "Enter number of hidden layers: ";
    cin >> hidden_layers;

    cout << "Enter the expected values:\n";
    vector<vector<float>> expected_value(g.num_nodes,vector<float>(out_dim));
    for(int i = 0; i < g.num_nodes; i++) {
        for(int j = 0; j < out_dim; j++) {
            cin >> expected_value[i][j];
        }
    }

    int hidden_dim=g.num_node_features;
    bool test_first_epoch=true;
    vector_Layer vl(hidden_layers,g.num_nodes,out_dim,hidden_dim);
    if(test_first_epoch) {
        for (auto& layer : vl.network_layer) {
            for (int i = 0; i < layer.input_dim; i++) {
                for (int j = 0; j < layer.output_dim; j++) {
                    layer.weight_matrix[i][j] = 1.0f;
                }
            }
        }
    } 

    for(int epoch_index = 1; epoch_index <= epochs; epoch_index++) {
        vl.forward_pass(out_dim,hidden_dim,hidden_layers,g);
        auto out_features = vl.network_layer[hidden_layers+1].layer_features;
        cout << "=== Node Features after epoch " << epoch_index <<  "===\n";
        for (size_t i = 0; i < out_features.size(); ++i) {
            cout << "Node " << i << ": ";
            for (float val : out_features[i]) {
                cout << val << " ";
            }
            cout << "\n";
        }
        cout << "\n";
        vl.backward_pass(learning_rate,expected_value);

        /* auto out_features = vl.network_layer[hidden_layers+1].layer_features;



        cout << "=== Node Features after epoch " << epoch_index <<  "===\n";
        for (size_t i = 0; i < out_features.size(); ++i) {
            cout << "Node " << i << ": ";
            for (float val : out_features[i]) {
                cout << val << " ";
            }
            cout << "\n";
        }
        cout << "\n"; */
        test_first_epoch=false;
    }    
    

    /* // 4) Compute node‐level scores (sum of features)
    vector<float> nodeScores;
    nodeScores.reserve(out_features.size());
    for (auto &feat : out_features) {
        float sum = accumulate(feat.begin(), feat.end(), 0.0f);
        nodeScores.push_back(sum);
    }

    // 5) Define CUSTOM edge‐combiner and graph‐aggregator
    // Example edge combiner: squared difference of endpoint scores
    OutputConverter::EdgeCombiner customEdgeCombiner = 
        [](float a, float b) {
            float diff = a - b;
            return diff * diff;
        };

    // Example graph aggregator: maximum node score
    OutputConverter::GraphAggregator customGraphAgg = 
        [](const OutputConverter::NodeScores& v) {
            if (v.empty()) return 0.0f;
            return *max_element(v.begin(), v.end());
        };

    // 6) Use OutputConverter with CUSTOM functions
    auto edgeScores = OutputConverter::toEdgeScores(
        nodeScores, 
        g,   
        OutputConverter::DefaultAgg::prodCombiner,// use squared‐difference combiner
        true                   // undirected graph
    );

    auto edgeTruth = OutputConverter::toEdgeBinary(
        nodeScores,
        g,
        0.5f,                  // threshold  
        OutputConverter::DefaultAgg::prodCombiner,// same combiner
        true
    );

    float graphScore = OutputConverter::toGraphScore(
        nodeScores, 
        OutputConverter::DefaultAgg::meanGraph        // use max aggregator
    );

    bool graphTruth = OutputConverter::toGraphBinary(
        nodeScores,
        0.5f,                  // threshold        
        OutputConverter::DefaultAgg::meanGraph  // same aggregator
    );

    // 7) Print results
    cout << "=== Graph‐Level ===\n";
    cout << "Score = " << graphScore
              << "\n\n";

    cout << "=== Edge‐Level ===\n";
    for (size_t i = 0; i < edgeScores.size(); ++i) {
        cout << "Edge " << i
                  << " | score = " << edgeScores[i]
                  << "\n";
    }
 */
    return 0;
}
