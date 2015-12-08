#pragma once
#include "Eigen/Dense"
#include "WordVec.h"
#include "Utils.h"
#include "Config.h"
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

	RowVectorXd input_layer;
	RowVectorXd hidden_layer;
	RowVectorXd global_input_layer;
	RowVectorXd global_hidden_layer;

	int word_dim, hidden_dim, window_size, neg_sample;

	GCWE() {};

	GCWE(int word_dim, int hidden_dim, int window_size, int neg_sample);

	double forward(WordVec& word_vec, RowVectorXi x,  RowVectorXi x_g);

	vector<MatrixXd> backward(WordVec& word_vec, RowVectorXi x, RowVectorXi x_g);

	void saveModel(string save_file);

	void loadModel(string model_file);
};

void train(Config conf, GCWE& gcwe_model, WordVec& word_vec, string src_raw_file, double learning_rate, int epoch, int branch_size, int window_size);

vector<MatrixXd> trainOneSentence(GCWE& gcwe_model, WordVec& word_vec, string sentence, int window_size, double learning_rate);

RowVectorXi getWindow(WordVec word_vec, string sentence, int window_size, int word_pos);

