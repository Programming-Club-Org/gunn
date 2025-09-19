#include <iostream>
#include "GraphReader.h"
#include "GCNTest.h" // or GraphSAGELayer, GATLayer

int main() {
    Graph g = read_graph_from_file("graph_data.txt");
    int out;
    std::cin >> out;
    GCNTestLayer gcn(g.num_node_features, out);
    gcn.forward(g.node_features, g.adjacency_list);
    auto updated_features = gcn.layer_features;

    for (const auto& node : updated_features) {
        for (float val : node) {
            std::cout << val << " ";
        }
        std::cout << "\n";
    }
}
