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

	input_layer = RowVectorXd(window_size*word_dim);
	hidden_layer = RowVectorXd(hidden_dim);
	global_input_layer = RowVectorXd(2 * word_dim);
	global_hidden_layer = RowVectorXd(hidden_dim);
}

double GCWE::forward(WordVec word_vec, RowVectorXi x, RowVectorXi x_g)
{
	for (int i = 0; i < window_size; i++)
	{
		for(int j = 0; j < word_dim; j++)
		{
			input_layer(i*word_dim + j) = word_vec.word_emb.row(x[i])(j);
		}
	}

	hidden_layer = tanh(input_layer * W1 + b1);

	double score_local = (hidden_layer * W2 + b2)(0);

	RowVectorXd global_info = RowVectorXd::Zero(word_dim);
	double sum_of_idf = 0;

	for (int w = 0; w < x_g.cols(); w++)
	{
		global_info += (word_vec.word_emb.row(x_g(w)) * word_vec.m_id_idf[x_g(w)]);
		sum_of_idf += word_vec.m_id_idf[x_g(w)];
	}
	global_info /= sum_of_idf;

	for(int j = 0; j < word_dim; j++)
	{
		global_input_layer(j) = word_vec.word_emb.row(x[window_size - 1])(j);
		global_input_layer(j+word_dim) = global_info(j);
	}

	global_hidden_layer = tanh(global_input_layer * Wg1 + bg1);

	double score_global = (global_hidden_layer * Wg2 + bg2)(0);

	double score = score_global + score_local;

	return score;
}

void GCWE::backward(WordVec word_vec, RowVectorXi x, RowVectorXi x_g)
{
	double pos_score = forward(word_vec, x, x_g);

	RowVectorXd pos_input_layer = input_layer;
	RowVectorXd pos_hidden_layer = hidden_layer;
	RowVectorXd pos_global_input_layer = global_input_layer;
	RowVectorXd pos_global_hidden_layer = global_hidden_layer;
	double sum_of_idf = 0;

	for (int w = 0; w < x_g.cols(); w++)
	{
		sum_of_idf += word_vec.m_id_idf[x_g(w)];
	}

	//derivation items
	MatrixXd dword_emb = MatrixXd::Zero(word_vec.word_emb.rows(), word_vec.word_emb.cols());
	MatrixXd dW2;
	MatrixXd dW1;
	RowVectorXd db1;
	MatrixXd dWg2;
	MatrixXd dWg1;
	RowVectorXd dbg1;

	srand(time(0));

	for(int i = 0; i < neg_sample; i++)
	{
		int neg_word = rand()%word_vec.vocb_size;

		RowVectorXi neg_seq = x;
		neg_seq(window_size-1) = neg_word;

		double neg_score = forward(word_vec, neg_seq, x_g);

		RowVectorXd neg_input_layer = input_layer;
		RowVectorXd neg_hidden_layer = hidden_layer;
		RowVectorXd neg_global_input_layer = global_input_layer;
		RowVectorXd neg_global_hidden_layer = global_hidden_layer;

		double f_error = (1-pos_score+neg_score > 0)?(1-pos_score+neg_score):0;

		//derivation for local network
		dW2 = neg_hidden_layer - pos_hidden_layer;

		dW1 = pos_input_layer.transpose() * mulByElem(-1 * W2.transpose(), derTanh(pos_hidden_layer)) +
			neg_input_layer.transpose() * mulByElem(W2.transpose(), derTanh(neg_hidden_layer));

		db1 = mulByElem(-1 * W2.transpose(), derTanh(pos_hidden_layer)) +
			mulByElem(W2.transpose(), derTanh(neg_hidden_layer));

		RowVectorXd dpos_input_layer = (W1 * mulByElem(-1 * W2.transpose(), derTanh(pos_hidden_layer)).transpose()).transpose();
		RowVectorXd dneg_input_layer = (W1 * mulByElem(W2.transpose(), derTanh(neg_hidden_layer)).transpose()).transpose();
		
		for (int i = 0; i < window_size - 1; i++)
		{
			dword_emb.row(x[i]) += dpos_input_layer.segment(i*word_dim, word_dim);
			dword_emb.row(x[i]) += dneg_input_layer.segment(i*word_dim, word_dim);
		}

		dword_emb.row(x[window_size-1]) += dpos_input_layer.segment((window_size-1)*word_dim, word_dim);
		dword_emb.row(neg_word) += dneg_input_layer.segment((window_size - 1)*word_dim, word_dim);

		//derivation for global network
		dWg2 = neg_global_hidden_layer - pos_global_hidden_layer;

		dWg1 = pos_global_input_layer.transpose() * mulByElem(-1 * Wg2.transpose(), derTanh(pos_global_hidden_layer)) +
			neg_global_input_layer.transpose() * mulByElem(Wg2.transpose(), derTanh(neg_global_hidden_layer));

		dbg1 = mulByElem(-1 * Wg2.transpose(), derTanh(pos_global_hidden_layer)) +
			mulByElem(Wg2.transpose(), derTanh(neg_global_hidden_layer));

		RowVectorXd dpos_global_input_layer = (Wg1 * mulByElem(-1 * Wg2.transpose(), derTanh(pos_global_hidden_layer)).transpose()).transpose();
		RowVectorXd dneg_global_input_layer = (Wg1 * mulByElem(Wg2.transpose(), derTanh(neg_global_hidden_layer)).transpose()).transpose();

		dword_emb.row(x[window_size - 1]) += dpos_global_input_layer.segment(0, word_dim);
		dword_emb.row(neg_word) += dneg_global_input_layer.segment(0, word_dim);

		for (int i = 0; i < x_g.cols(); i++)
		{
			dword_emb.row(x_g(i)) += (dpos_global_input_layer.segment(word_dim, word_dim)*word_vec.m_id_idf[x_g(i)]/sum_of_idf);
			dword_emb.row(x_g(i)) += (dneg_global_input_layer.segment(word_dim, word_dim)*word_vec.m_id_idf[x_g(i)] / sum_of_idf);
		}
	}

	word_vec.word_emb += dword_emb;
	W1 += dW1;
	b1 += db1;
	W2 += dW2;
	Wg1 += dWg1;
	bg1 += dbg1;
	Wg2 += dWg2;
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

	RowVectorXi x_g = x;

	gcwe_model.backward(word_vec, x, x_g);

	return 0;
}
