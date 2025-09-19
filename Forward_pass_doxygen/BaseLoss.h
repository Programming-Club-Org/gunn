#pragma once
#include <vector>
#include <cmath>
using namespace std;

class BaseLoss {
public:

    float loss_value; // loss value

    vector<vector<float>> CrossEntropyLoss(
        vector<vector<float>> output_features,
        vector<vector<float>> expected_value
    ) {
        vector<vector<float>> grad_CEL;
        int n = output_features.size();
        int o_features=output_features[0].size();
        vector<vector<float>> softmax;
        for(int i =0; i < n; i++) {
            float exp_sum=0;
            for(int j = 0; j < o_features; j++) {
                exp_sum += exp(output_features[i][j]);
            }
            for(int k = 0; k < o_features; k++) {
                softmax[i][k]=(exp(output_features[i][k]))/exp_sum;
            }
        }
        loss_value=0;
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < o_features; j++) {
                loss_value+=(-1)*expected_value[i][j]*log(softmax[i][j]);
            }
        }
        loss_value=loss_value/n; //taking average over all nodes
        for(int i =0; i < n; i++) {
            for(int j = 0; j < o_features; j++) {
                grad_CEL[i][j]=softmax[i][j]-expected_value[i][j];
            }
        }
    }
};