#include "BaseLoss.h"
#include <cmath>
#include <limits>
#include <algorithm>
#include <iostream>

vector<vector<float>> BaseLoss::CrossEntropyLoss(
        vector<vector<float>> output_features,
        vector<vector<float>> expected_value
    ) {
        int n = output_features.size();
        if(n == 0) {return {};}
        int o_features=output_features[0].size();
        vector<vector<float>> softmax(n,vector<float>(o_features,0.0f));
        vector<vector<float>> grad_CEL(n,vector<float>(o_features,0.0f));
        for(int i =0; i < n; i++) {
            float exp_sum=0.0f;
            for(int j = 0; j < o_features; j++) {
                exp_sum += exp(output_features[i][j]);
            }
            if (exp_sum == 0.0) exp_sum = numeric_limits<float>::min();
            for(int k = 0; k < o_features; k++) {
                softmax[i][k]=static_cast<float>(exp(output_features[i][k]))/exp_sum;
            }
        }
        loss_value=0.0f;
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < o_features; j++) {
                float p=softmax[i][j];
                if (p<numeric_limits<float>::min()) {p = numeric_limits<float>::min();}
                loss_value+=(-1)*expected_value[i][j]*log(p);
            }
        }
        loss_value=loss_value/static_cast<float>(n); //taking average over all nodes
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < o_features; j++) {
                grad_CEL[i][j]=(softmax[i][j]-expected_value[i][j])/static_cast<float>(n);
            }
        }
        return grad_CEL;
    }