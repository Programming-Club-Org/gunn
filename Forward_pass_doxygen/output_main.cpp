// main.cpp

#include "Graph.h"              // your Graph class
#include "GraphReader.h"        // read_graph_from_file(...)
#include "GCNTest.h"               // your existing GCNLayer
#include "output.h"             // OutputConverter API
#include <iostream>
#include <vector>
#include <numeric>
#include <functional>           // for function

int main(int argc, char** argv) {
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
    int out_dim,hidden_layers;
    cout << "Enter number of hidden layers: ";
    cin >> out_dim >> hidden_layers;

    vector<vector<float>> aggregated_features(g.num_nodes);

     // Precompute degrees
    vector<int> degrees(g.num_nodes);
    for (int i = 0; i < g.num_nodes; i++) {
        degrees[i] = g.adjacency_list[i].size();
    }

    int hidden_features=g.num_node_features;
    GCNTestLayer gcn_in(g.num_node_features,hidden_features);
    for(int i = 0; i < g.num_nodes; i++ ) {
        vector<float> aggregated = gcn_in.aggregate_neighbors(i, g.node_features, g.adjacency_list, degrees);
        aggregated_features.push_back(aggregated);
    }

    
    auto curr_features=aggregated_features;
    vector<vector<float>> out_features;

    for(int layer = 1; layer <= hidden_layers; layer++ ) {
        GCNTestLayer gcn_hid(hidden_features,hidden_features);
        gcn_hid.forward(curr_features,g.adjacency_list);
        out_features=gcn_hid.layer_features;
        curr_features=out_features;
    }

    GCNTestLayer gcn_out(hidden_features,out_dim);
    gcn_out.forward(curr_features,g.adjacency_list);
    out_features=gcn_out.layer_features;


    cout << "=== Node Features (post-GCN) ===\n";
    for (size_t i = 0; i < out_features.size(); ++i) {
        cout << "Node " << i << ": ";
        for (float val : out_features[i]) {
            cout << val << " ";
        }
        cout << "\n";
    }
    cout << "\n";

    // 4) Compute node‐level scores (sum of features)
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

    return 0;
}
