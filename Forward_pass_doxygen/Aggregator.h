// Aggregator.h
#pragma once

#include "Graph.h"
#include <vector>
#include <functional>
#include <numeric>
#include <algorithm>
#include <cmath>

namespace OutputConverter {
  /**
 * @namespace OutputConverter
 * @brief Converters from node‐level scores to edge‐ or graph‐level metrics.
 */   
  //──────────────────────────────────────────────────────────────────────────
  // Type aliases matching OutputConverter signatures
  //──────────────────────────────────────────────────────────────────────────

  /**  A vector of per‐node scalar scores*/
  using NodeScores   = std::vector<float>;

  /** A vector of per‐edge scalar scores*/
  using EdgeScores   = std::vector<float>;

  /** A vector of boolean flags*/
  using BinaryVector = std::vector<bool>;

  /** Combines two node‐scores into one edge‐score*/
  using EdgeCombiner    = std::function<float(float, float)>;

  /** Aggregates all node‐scores into one graph‐score*/
  using GraphAggregator = std::function<float(const NodeScores&)>;


  //──────────────────────────────────────────────────────────────────────────
  // Default implementations (used when the user omits their own)
  //──────────────────────────────────────────────────────────────────────────
  namespace DefaultAgg {
  /**
 * @namespace OutputConverter::DefaultAgg
 * @brief Default combiner and aggregator functions.
 */

    // sum of endpoint scores
    /**
   * @brief Sum of two endpoint scores.
   * @param a First endpoint score.
   * @param b Second endpoint score.
   * @return The value a + b.
   */
    inline float sumCombiner(float a, float b) {
      return a + b;
    }

    // product of endpoint scores
    /**
   * @brief Product of two endpoint scores.
   * @param a First endpoint score.
   * @param b Second endpoint score.
   * @return The value a * b.
   */
    inline float prodCombiner(float a, float b) {
      return a * b;
    }

    // maximum of endpoint scores
    /**
   * @brief Maximum of two endpoint scores.
   * @param a First endpoint score.
   * @param b Second endpoint score.
   * @return The greater of a and b.
   */
    inline float maxCombiner(float a, float b) {
      return std::max(a, b);
    }

    // minimum of endpoint scores

  /**
   * @brief Minimum of two endpoint scores.
   * @param a First endpoint score.
   * @param b Second endpoint score.
   * @return The lesser of a and b.
   */
    inline float minCombiner(float a, float b) {
      return std::min(a, b);
    }

    // absolute difference of endpoint scores
    /**
   * @brief Absolute difference of two endpoint scores.
   * @param a First endpoint score.
   * @param b Second endpoint score.
   * @return The value |a - b|.
   */
    inline float absDiffCombiner(float a, float b) {
      return std::fabs(a - b);
    }

    // sum of all node scores
    /**
   * @brief Sum of all node scores in a graph.
   * @param scores Vector of node scores.
   * @return The total sum of scores.
   */
    inline float sumGraph(const NodeScores& scores) {
      return std::accumulate(scores.begin(), scores.end(), 0.0f);
    }

    // mean of all node scores
    /**
   * @brief Mean of all node scores in a graph.
   * @param scores Vector of node scores.
   * @return The arithmetic mean, or 0.0 if empty.
   */
    inline float meanGraph(const NodeScores& scores) {
      if (scores.empty()) return 0.0f;
      return std::accumulate(scores.begin(), scores.end(), 0.0f)
           / static_cast<float>(scores.size());
    }

    // maximum of all node scores
    /**
   * @brief Maximum node score in a graph.
   * @param scores Vector of node scores.
   * @return The largest score, or 0.0 if empty.
   */
    inline float maxGraph(const NodeScores& scores) {
      if (scores.empty()) return 0.0f;
      return *std::max_element(scores.begin(), scores.end());
    }

    // minimum of all node scores
  /**
   * @brief Minimum node score in a graph.
   * @param scores Vector of node scores.
   * @return The smallest score, or 0.0 if empty.
   */
    inline float minGraph(const NodeScores& scores) {
      if (scores.empty()) return 0.0f;
      return *std::min_element(scores.begin(), scores.end());
    }

  }  // namespace DefaultAgg

}  // namespace OutputConverter
