#ifndef GRAPH_H
#define GRAPH_H

#include <vector>
using namespace std;
// Produces a clear abstraction for representing a graph structure
// with node features, adjacency list, optionally edge and global features
// to facilitate GNN architecture
/**
 * @class Graph
 * @brief Represents a graph with node, edge, and features.
 */
class Graph {
public:
    int num_nodes;         /** Number of nodes in the graph*/
    int num_node_features; /** Number of features per node*/

    vector<vector<float>> node_features;   /** Feature matrix : [n_nodes][n_node_features]*/
    vector<vector<int>> adjacency_list;    /** graph_representation : [n_nodes][variable number of neighbors]*/

    // Optional features :
    vector<vector<float>> edge_features;   /** For edge-conditioned GNNs*/
    vector<float> global_features;         /** For graph-level attributes*/

    /** Stores (src, dst) for every edge added*/
    vector<pair<int, int>> edge_list;

    // Constructs a graph with the specified number of nodes and node feature dimensions.
    // Initializes empty adjacency list and zero-initialized feature matrices.
    /**
     * @brief Constructs a Graph with specified nodes and feature dimensions.
     * @param num_nodes Number of nodes to allocate.
     * @param num_node_features Dimension of each node’s feature vector.
     * @details
     * Initializes `node_features` to zeros and sets up an empty adjacency list.
     */
    Graph(int num_nodes, int num_node_features);

    // Move constructor for efficient transfers without deep copying
    /**
     * @brief Move constructor for efficient ownership transfer.
     * @param other Graph instance to move from.
     */
    Graph(Graph&& other) noexcept;

    // Move assignment operator for efficient transfers without deep copying
    /**
     * @brief Move assignment operator for efficient ownership transfer.
     * @param other Graph instance to move from.
     * @return Reference to this graph after move.
     */
    Graph& operator=(Graph&& other) noexcept;

    // Adds an undirected edge between source node and destination node
    /**
     * @brief Adds an undirected edge between two nodes.
     * @param src Source node index.
     * @param dst Destination node index.
     * @details
     * Appends `dst` to the neighbor list of `src` and vice versa,
     * then records the edge in `edge_list`.
     */
    void add_edge(int src, int dst);

    // Sets the feature vector of a specific node
    /**
     * @brief Sets the feature vector for a given node.
     * @param node_id Index of the node whose features are being set.
     * @param features Vector of floats, length must equal `num_node_features`.
     * @details
     * Overwrites the existing feature row if the size matches.
     * Behavior is undefined if `features.size() != num_node_features`.
     */
    void set_node_feature(int node_id, const vector<float>& features);

    // New accessor: returns the src and dst for a given edge ID
    /**
     * @brief Retrieves the edge at a specific index.
     * @param edge_id Index into the `edge_list`.
     * @return A `std::pair<int, int>` containing (src, dst) of the edge.
     */
    pair<int, int> edge(size_t edge_id) const;
};

#endif
