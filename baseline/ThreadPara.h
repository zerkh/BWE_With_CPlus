#pragma once
#include "Utils.h"
#include "GCWE.h"
#include "TE.h"

class GCWEThread
{
public:
	MatrixXd dword_emb;
	MatrixXd dW1;
	RowVectorXd db1;
	MatrixXd dW2;
	RowVectorXd db2;
	MatrixXd dWg1;
	RowVectorXd dbg1;
	MatrixXd dWg2;
	RowVectorXd dbg2;

	int word_dim;
	int hidden_dim;
	int window_size;
	int batch_size;
	double learning_rate;
	vector<string> sentences;
	GCWE* gcwe_model;
	WordVec* word_vec;

	GCWEThread() {};

	void init(GCWE& gcwe, WordVec& word_vec, int word_dim, int hidden_dim, int window_size, double learning_rate)
	{
		dword_emb = MatrixXd::Zero(word_vec.vocb_size, word_dim);

		dW1 = MatrixXd::Zero(window_size*word_dim, hidden_dim);
		db1 = RowVectorXd::Zero(hidden_dim);
		dW2 = MatrixXd::Zero(hidden_dim, 1);
		db2 = RowVectorXd::Zero(1);

		dWg1 = MatrixXd::Zero(2 * word_dim, hidden_dim);
		dbg1 = RowVectorXd::Zero(hidden_dim);
		dWg2 = MatrixXd::Zero(hidden_dim, 1);
		dbg2 = RowVectorXd::Zero(1);

		gcwe_model = &gcwe;
		this->word_vec = &word_vec;

		this->learning_rate = learning_rate;
		this->word_dim = word_dim;
		this->hidden_dim = hidden_dim;
		this->window_size = window_size;
	}

	void clear()
	{
		dword_emb = MatrixXd::Zero(word_vec->vocb_size, word_dim);

		dW1 = MatrixXd::Zero(window_size*word_dim, hidden_dim);
		db1 = RowVectorXd::Zero(hidden_dim);
		dW2 = MatrixXd::Zero(hidden_dim, 1);
		db2 = RowVectorXd::Zero(1);

		dWg1 = MatrixXd::Zero(2 * word_dim, hidden_dim);
		dbg1 = RowVectorXd::Zero(hidden_dim);
		dWg2 = MatrixXd::Zero(hidden_dim, 1);
		dbg2 = RowVectorXd::Zero(1);
	}
};

class TEThread
{
public:
	MatrixXd dword_emb;
	MatrixXd dW1;
	RowVectorXd db1;
	MatrixXd dW2;
	RowVectorXd db2;
	MatrixXd dWg1;
	RowVectorXd dbg1;
	MatrixXd dWg2;
	RowVectorXd dbg2;

	int word_dim;
	int hidden_dim;
	int window_size;
	int batch_size;
	double learning_rate;
	double lambda;

	MatrixXd alignTable;
	vector<string> sentences;
	GCWE* gcwe_model;
	WordVec* src_word_vec;
	WordVec* tgt_word_vec;
	TE* te_model;

	TEThread() {};

	void init(GCWE& gcwe, TE& te, WordVec& src_word_vec, WordVec& tgt_word_vec, int word_dim, int hidden_dim, int window_size, double learning_rate, double lambda)
	{
		dword_emb = MatrixXd::Zero(tgt_word_vec.vocb_size, word_dim);

		dW1 = MatrixXd::Zero(window_size*word_dim, hidden_dim);
		db1 = RowVectorXd::Zero(hidden_dim);
		dW2 = MatrixXd::Zero(hidden_dim, 1);
		db2 = RowVectorXd::Zero(1);

		dWg1 = MatrixXd::Zero(2 * word_dim, hidden_dim);
		dbg1 = RowVectorXd::Zero(hidden_dim);
		dWg2 = MatrixXd::Zero(hidden_dim, 1);
		dbg2 = RowVectorXd::Zero(1);

		gcwe_model = &gcwe;
		this->src_word_vec = &src_word_vec;
		this->tgt_word_vec = &tgt_word_vec;
		te_model = &te;

		this->lambda = lambda;
		this->learning_rate = learning_rate;
		this->word_dim = word_dim;
		this->hidden_dim = hidden_dim;
		this->window_size = window_size;
	}

	void clear()
	{
		dword_emb = MatrixXd::Zero(tgt_word_vec->vocb_size, word_dim);

		dW1 = MatrixXd::Zero(window_size*word_dim, hidden_dim);
		db1 = RowVectorXd::Zero(hidden_dim);
		dW2 = MatrixXd::Zero(hidden_dim, 1);
		db2 = RowVectorXd::Zero(1);

		dWg1 = MatrixXd::Zero(2 * word_dim, hidden_dim);
		dbg1 = RowVectorXd::Zero(hidden_dim);
		dWg2 = MatrixXd::Zero(hidden_dim, 1);
		dbg2 = RowVectorXd::Zero(1);
	}
};