#ifndef GRAPH_DATA_READER_H
#define GRAPH_DATA_READER_H

#include "Graph.h"
#include <string>
using namespace std;
/**
 * @brief Reads input data from text file and creates a graph object out of it
 * @return A graph class object, created out of the input data
 * @param filename Name of the file with the input data
 * @details The format for the input data should be : 
 * -number of nodes
 * -number of features
 * -node features : number of the node followed by all it's features
 * -edge list
 */
Graph read_graph_from_file(const string& filename);

#endif
