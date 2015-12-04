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
	VectorXd b1;
	MatrixXd W2;
	VectorXd b2;
	MatrixXd Wg1;
	VectorXd bg1;
	MatrixXd Wg2;
	VectorXd bg2;
	int word_dim, hidden_dim, window_size;

	GCWE(int word_dim, int hidden_dim, int window_size);

	int forward(WordVec word_vec, VectorXd x,  VectorXd x_g);

	void backward();
};
