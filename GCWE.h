#pragma once
#include "Eigen/Core"
#include "WordVec.h"
#include <iostream>
using namespace std;
using namespace Eigen;

/*
Reference Huang et al. 2012 Global-Context Word Embedding
*/
class GCWE
{
public:
	MatrixXd W1;
	RowVectorXd b1;
	MatrixXd W2;
	RowVectorXd b2;
	MatrixXd Wg1;
	RowVectorXd bg1;
	MatrixXd Wg2;
	RowVectorXd bg2;
	int word_dim, hidden_dim, window_size, neg_sample;

	GCWE(int word_dim, int hidden_dim, int window_size, int neg_sample);

	double forward(WordVec word_vec, RowVectorXi x,  RowVectorXd x_g);

	void backward(WordVec word_vec, RowVectorXi x, RowVectorXd x_g);
};
