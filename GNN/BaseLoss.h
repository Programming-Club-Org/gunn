#pragma once
#include <vector>
using namespace std;

class BaseLoss {
public:

    float loss_value; // loss value

    vector<vector<float>> CrossEntropyLoss(
        vector<vector<float>> output_features,
        vector<vector<float>> expected_value
    );
};