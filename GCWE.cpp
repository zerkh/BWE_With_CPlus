#include "Eigen/Core"
#include "GCWE.h"
#include "Utils.h"
#include <iostream>
using namespace std;
using namespace Eigen;

GCWE::GCWE(int word_dim, int hidden_dim, int window_size)
{
	this->word_dim = word_dim;
	this->hidden_dim = hidden_dim;
	this->window_size = window_size;

	W1 = MatrixXd::Random(window_size*word_dim, hidden_dim);
	b1 = VectorXd::Zero(hidden_dim);
	W2 = MatrixXd::Random(hidden_dim, 1);
	b2 = VectorXd::Zero(1, 1);

	Wg1 = MatrixXd::Random(2 * word_dim, hidden_dim);
	bg1 = VectorXd::Zero(hidden_dim);
	Wg2 = MatrixXd::Random(hidden_dim, 1);
	bg2 = VectorXd::Zero(1, 1);
}

int GCWE::forward(WordVec word_vec, VectorXd x, VectorXd x_g)
{
	VectorXd input_layer(word_dim * window_size);
	for (int i = 0; i < window_size; i++)
	{
		input_layer << word_vec.word_emb.row(x(i));
	}

	VectorXd hidden_layer = tanh(input_layer * W1 + b1);

	int score_local = (hidden_layer * W2 + b2)(0);

	VectorXd global_input_layer(word_dim * 2);
	global_input_layer << word_vec.word_emb.row(x[window_size - 1]), x_g;

	VectorXd global_hidden_layer = tanh(global_input_layer * Wg1 + bg1);

	int score_global = (global_hidden_layer * Wg2 + bg2)(0);

	int score = score_global + score_local;

	return score;
}

void GCWE::backward()
{

}

