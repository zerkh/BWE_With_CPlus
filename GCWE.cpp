#include "Eigen/Core"
#include "GCWE.h"
#include "Utils.h"
#include <iostream>
using namespace std;
using namespace Eigen;

GCWE::GCWE(int word_dim, int hidden_dim, int window_size, int neg_sample)
{
	this->word_dim = word_dim;
	this->hidden_dim = hidden_dim;
	this->window_size = window_size;
	this->neg_sample = neg_sample;

	W1 = MatrixXd::Random(window_size*word_dim, hidden_dim);
	b1 = RowVectorXd::Zero(hidden_dim);
	W2 = MatrixXd::Random(hidden_dim, 1);
	b2 = RowVectorXd::Zero(1);

	Wg1 = MatrixXd::Random(2 * word_dim, hidden_dim);
	bg1 = RowVectorXd::Zero(hidden_dim);
	Wg2 = MatrixXd::Random(hidden_dim, 1);
	bg2 = RowVectorXd::Zero(1);
}

double GCWE::forward(WordVec word_vec, RowVectorXi x, RowVectorXd x_g)
{
	RowVectorXd input_layer(word_dim * window_size);
	for (int i = 0; i < window_size; i++)
	{
		for(int j = 0; j < word_dim; j++)
		{
			input_layer(i*word_dim + j) = word_vec.word_emb.row(x[i])(j);
		}
	}

	RowVectorXd hidden_layer = tanh(input_layer * W1 + b1);

	double score_local = (hidden_layer * W2 + b2)(0);

	RowVectorXd global_input_layer(word_dim * 2);
	for(int j = 0; j < word_dim; j++)
	{
		global_input_layer(j) = word_vec.word_emb.row(x[window_size - 1])(j);
		global_input_layer(j+word_dim) = x_g(j);
	}

	RowVectorXd global_hidden_layer = tanh(global_input_layer * Wg1 + bg1);

	double score_global = (global_hidden_layer * Wg2 + bg2)(0);

	double score = score_global + score_local;

	return score;
}

void GCWE::backward(WordVec word_vec, RowVectorXi x, RowVectorXd x_g)
{
	double pos_score = forward(word_vec, x, x_g);

	srand(time(0));

	for(int i = 0; i < neg_sample; i++)
	{
		int neg_word = rand()%word_vec.vocb_size;

		RowVectorXi neg_seq = x;
		neg_seq(window_size-1) = neg_word;

		double neg_score = forward(word_vec, neg_seq, x_g);

		double f_error = (1-pos_score+neg_score > 0)?(1-pos_score+neg_score):0;

		cout << f_error << endl;
	}
}

int main()
{
	WordVec word_vec(25, 10);
	GCWE gcwe_model(25, 50, 5, 5);

	RowVectorXi x(5);
	for(int i = 0; i < 5; i++)
	{
		x(i) = i;
	}

	RowVectorXd x_g = RowVectorXd::Random(25);

	gcwe_model.backward(word_vec, x, x_g);

	return 0;
}
