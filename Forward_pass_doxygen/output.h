#ifndef OUTPUT_CONVERTER_H
#define OUTPUT_CONVERTER_H

#include <vector>
#include <functional>
#include "Graph.h"       // your Graph class
#include "Aggregator.h"  // brings in NodeScores, EdgeCombiner, GraphAggregator, DefaultAgg

using namespace std;

namespace OutputConverter {
  /** A vector of per‐node scalar scores. */
  using NodeScores   = vector<float>;   
  /** A vector of per‐edge scalar scores. */
  using EdgeScores   = vector<float>;   
  /**
   * @brief A vector of boolean labels.
   * @details Used for thresholded edge or graph outputs.
   */
  using BinaryVector = vector<bool>;     

  // pull in our aliases
  /**
   * @brief Combines two node scores into a single edge score.
   * @details You can supply your own combiner or use one of the defaults
   *          (e.g., DefaultAgg::prodCombiner, DefaultAgg::sumCombiner).
   */
  using EdgeCombiner    = ::OutputConverter::EdgeCombiner;
  /**
   * @brief Aggregates node scores into one graph‐level score.
   * @details You can supply your own aggregator or use a default
   *          (e.g., DefaultAgg::meanGraph, DefaultAgg::sumGraph).
   */
  using GraphAggregator = ::OutputConverter::GraphAggregator;

  // If the user omits a combiner/aggregator, we default to these:
  //   DefaultAgg::prodCombiner   → a * b   (matches your old default)
  //   DefaultAgg::meanGraph      → mean(v) (matches your old default)
  /**
 * @brief Compute a score for each edge in the graph.
 * @param nodeScores  Vector of scores, one per node (size = graph.num_nodes).
 * @param graph Graph instance whose edges define which node pairs to combine.
 * @param combiner Callable f(a,b) to merge two node scores into an edge score.
 * Defaults to DefaultAgg::prodCombiner.
 * @param undirected  If true, treat edges as undirected (include each pair once).
 * Defaults to true.
 * @return EdgeScores Vector of computed edge scores.
 */
  EdgeScores toEdgeScores(
    const NodeScores& nodeScores,
    const Graph&      graph,
    EdgeCombiner      combiner   = DefaultAgg::prodCombiner,
    bool              undirected = true
  );

/**
 * @brief Convert node scores into binary edge labels.
 * @param nodeScores Vector of scores, one per node.
 * @param graph  Graph instance.
 * @param threshold Scalar threshold: edge score > threshold yields true.
 * @param combiner Callable to combine two node scores; default as above.
 * @param undirected  If true, treat graph as undirected.
 * @return BinaryVector True/false label per edge.
 */
  BinaryVector toEdgeBinary(
    const NodeScores& nodeScores,
    const Graph&      graph,
    float             threshold,
    EdgeCombiner      combiner   = DefaultAgg::prodCombiner,
    bool              undirected = true
  );
/**
 * @brief Aggregate node scores into a single graph‐level score.
 * @param nodeScores Vector of scores, one per node.
 * @param aggregator Callable f(nodeScores) to compute a scalar summary.
 * Defaults to DefaultAgg::meanGraph.
 * @return Floating‐point graph‐level score.
 */
  float toGraphScore(
    const NodeScores&   nodeScores,
    GraphAggregator     aggregator = DefaultAgg::meanGraph
  );
/**
 * @brief Aggregate node scores into a single graph‐level score.
 * @param nodeScores Vector of scores, one per node.
 * @param aggregator Callable f(nodeScores) to compute a scalar summary.
 * Defaults to DefaultAgg::meanGraph.
 * @return Floating‐point graph‐level score.
 */
  bool toGraphBinary(
    const NodeScores&   nodeScores,
    float               threshold,
    GraphAggregator     aggregator = DefaultAgg::meanGraph
  );

} // namespace OutputConverter

#endif // OUTPUT_CONVERTER_H
