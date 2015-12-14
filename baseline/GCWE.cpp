#include "GCWE.h"
#include "Utils.h"
#include "Config.h"
#include "ThreadPara.h"
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

double GCWE::forward(WordVec& word_vec, RowVectorXi x, RowVectorXi x_g)
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
		global_info += (word_vec.word_emb.row(x_g(w)) * word_vec.v_id_idf[x_g(w)]);
		sum_of_idf += word_vec.v_id_idf[x_g(w)];
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

vector<MatrixXd> GCWE::backward(WordVec& word_vec, RowVectorXi x, RowVectorXi x_g)
{
	double pos_score = forward(word_vec, x, x_g);

	RowVectorXd pos_input_layer = input_layer;
	RowVectorXd pos_hidden_layer = hidden_layer;
	RowVectorXd pos_global_input_layer = global_input_layer;
	RowVectorXd pos_global_hidden_layer = global_hidden_layer;
	double sum_of_idf = 0;

	for (int w = 0; w < x_g.cols(); w++)
	{
		sum_of_idf += word_vec.v_id_idf[x_g(w)];
	}

	//derivation items
	MatrixXd s_dword_emb = MatrixXd::Zero(word_vec.word_emb.rows(), word_vec.word_emb.cols());
	MatrixXd s_dW2 = MatrixXd::Zero(W2.rows(), W2.cols());
	MatrixXd s_dW1 = MatrixXd::Zero(W1.rows(), W1.cols());
	RowVectorXd s_db1 = RowVectorXd::Zero(b1.cols());
	MatrixXd s_dWg2 = MatrixXd::Zero(Wg2.rows(), Wg2.cols());
	MatrixXd s_dWg1 = MatrixXd::Zero(Wg1.rows(), Wg1.cols());
	RowVectorXd s_dbg1 = RowVectorXd::Zero(bg1.cols());

	srand(time(0));

	for(int epoch = 0; epoch < neg_sample; epoch++)
	{
		int neg_word = rand()%word_vec.vocb_size;

		RowVectorXi neg_seq = x;
		neg_seq(window_size-1) = neg_word;
		MatrixXd dword_emb = MatrixXd::Zero(word_vec.word_emb.rows(), word_vec.word_emb.cols());

		double neg_score = forward(word_vec, neg_seq, x_g);

		RowVectorXd neg_input_layer = input_layer;
		RowVectorXd neg_hidden_layer = hidden_layer;
		RowVectorXd neg_global_input_layer = global_input_layer;
		RowVectorXd neg_global_hidden_layer = global_hidden_layer;

		double f_error = (1-pos_score+neg_score > 0)?(1-pos_score+neg_score):0;

		//derivation for local network
		MatrixXd dW2 = neg_hidden_layer.transpose() - pos_hidden_layer.transpose();

		MatrixXd dW1 = pos_input_layer.transpose() * mulByElem(-1 * W2.transpose(), derTanh(pos_hidden_layer)) +
			neg_input_layer.transpose() * mulByElem(W2.transpose(), derTanh(neg_hidden_layer));

		RowVectorXd db1 = mulByElem(-1 * W2.transpose(), derTanh(pos_hidden_layer)) +
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
		MatrixXd dWg2 = neg_global_hidden_layer.transpose() - pos_global_hidden_layer.transpose();

		MatrixXd dWg1 = pos_global_input_layer.transpose() * mulByElem(-1 * Wg2.transpose(), derTanh(pos_global_hidden_layer)) +
			neg_global_input_layer.transpose() * mulByElem(Wg2.transpose(), derTanh(neg_global_hidden_layer));

		RowVectorXd dbg1 = mulByElem(-1 * Wg2.transpose(), derTanh(pos_global_hidden_layer)) +
			mulByElem(Wg2.transpose(), derTanh(neg_global_hidden_layer));

		RowVectorXd dpos_global_input_layer = (Wg1 * mulByElem(-1 * Wg2.transpose(), derTanh(pos_global_hidden_layer)).transpose()).transpose();
		RowVectorXd dneg_global_input_layer = (Wg1 * mulByElem(Wg2.transpose(), derTanh(neg_global_hidden_layer)).transpose()).transpose();

		dword_emb.row(x[window_size - 1]) += dpos_global_input_layer.segment(0, word_dim);
		dword_emb.row(neg_word) += dneg_global_input_layer.segment(0, word_dim);

		for (int i = 0; i < x_g.cols(); i++)
		{
			dword_emb.row(x_g(i)) += (dpos_global_input_layer.segment(word_dim, word_dim)*word_vec.v_id_idf[x_g(i)]/sum_of_idf);
			dword_emb.row(x_g(i)) += (dneg_global_input_layer.segment(word_dim, word_dim)*word_vec.v_id_idf[x_g(i)] / sum_of_idf);
		}

		s_dword_emb += dword_emb;
		s_dW1 += dW1;
		s_db1 += db1;
		s_dW2 += dW2;
		s_dWg1 += dWg1;
		s_dbg1 += dbg1;
		s_dWg2 += dWg2;
	}

	//word_vec.word_emb += s_dword_emb;
	//W1 += s_dW1;
	//b1 += s_db1;
	//W2 += s_dW2;
	//Wg1 += s_dWg1;
	//bg1 += s_dbg1;
	//Wg2 += s_dWg2;

	vector<MatrixXd> derivations;
	derivations.push_back(s_dword_emb);
	derivations.push_back(s_dW1);
	derivations.push_back(s_db1);
	derivations.push_back(s_dW2);
	derivations.push_back(s_dWg1);
	derivations.push_back(s_dbg1);
	derivations.push_back(s_dWg2);

	return derivations;
}

void GCWE::saveModel(string save_file)
{
	ofstream out(save_file.c_str(), ios::out);

	out << "W1:" << endl;
	out << W1 << endl;

	out << "b1:" << endl;
	out << b1 << endl;

	out << "W2:" << endl;
	out << W2 << endl;

	out << "b2:" << endl;
	out << b2 << endl;

	out << "Wg1:" << endl;
	out << Wg1 << endl;

	out << "bg1:" << endl;
	out << bg1 << endl;

	out << "Wg2:" << endl;
	out << Wg2 << endl;

	out << "bg2:" << endl;
	out << bg2 << endl;

	out.close();
}

void GCWE::loadModel(string model_file)
{
	ifstream in(model_file.c_str(), ios::in);


	in.close();
}

