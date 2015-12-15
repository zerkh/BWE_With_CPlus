#ifndef __THREADPARA__
#define __THREADPARA__

#include "Utils.h"
#include "GCWE.h"
#include "TE.h"
#include "SkipGram.h"

class SkipGramThread
{
public:
	MatrixXd dword_emb;
	MatrixXd dW;

	int word_dim;
	int window_size;
	int batch_size;
	double learning_rate;
	vector<string> sentences;
	SkipGram skipgram_model;
	WordVec word_vec;

	SkipGramThread() {};

	void init(SkipGram skipgram, WordVec word_vec, int word_dim, int window_size, double learning_rate)
	{
		dword_emb = MatrixXd::Zero(word_vec.vocb_size, word_dim);

		dW = MatrixXd::Zero(word_dim, word_vec.vocb_size);

		skipgram_model = skipgram;
		this->word_vec = word_vec;

		this->learning_rate = learning_rate;
		this->word_dim = word_dim;
		this->window_size = window_size;
	}

	void update(SkipGram skipgram, WordVec word_vec)
	{
		this->skipgram_model = skipgram;
		this->word_vec = word_vec;
	}

	void clear()
	{
		dword_emb = MatrixXd::Zero(word_vec.vocb_size, word_dim);

		dW = MatrixXd::Zero(word_dim, word_vec.vocb_size);
	}
};

class TEThread
{
public:
	MatrixXd src_dword_emb;
	MatrixXd src_dW;
	MatrixXd tgt_dword_emb;
	MatrixXd tgt_dW;

	int word_dim;
	int hidden_dim;
	int window_size;
	int batch_size;
	double learning_rate;
	double lambda;

	MatrixXd alignTable;
	vector<string> src_sentences;
	vector<string> tgt_sentences;
	SkipGram src_skipgram_model;
	SkipGram tgt_skipgram_model;
	WordVec src_word_vec;
	WordVec tgt_word_vec;
	TE src_te_model;
	TE tgt_te_model;

	TEThread() {};

	void init(SkipGram src_skipgram, SkipGram tgt_skipgram,
				TE src_te, TE tgt_te,
				WordVec src_word_vec, WordVec tgt_word_vec,
				int word_dim, int window_size, double learning_rate, double lambda)
	{
		src_dword_emb = MatrixXd::Zero(src_word_vec.vocb_size, word_dim);
		src_dW = MatrixXd::Zero(word_dim, src_word_vec.vocb_size);

		tgt_dword_emb = MatrixXd::Zero(tgt_word_vec.vocb_size, word_dim);
		tgt_dW = MatrixXd::Zero(word_dim, tgt_word_vec.vocb_size);

		this->src_skipgram_model = src_skipgram;
		this->tgt_skipgram_model = tgt_skipgram;
		this->src_word_vec = src_word_vec;
		this->tgt_word_vec = tgt_word_vec;
		this->src_te_model = src_te;
		this->tgt_te_model = tgt_te;

		this->lambda = lambda;
		this->learning_rate = learning_rate;
		this->word_dim = word_dim;
		this->window_size = window_size;
	}

	void update(SkipGram src_skipgram_model, SkipGram tgt_skipgram_model,
				TE src_te_model, TE tgt_te_model,
				WordVec src_word_vec, WordVec tgt_word_vec)
	{
		this->src_skipgram_model = src_skipgram_model;
		this->tgt_skipgram_model = tgt_skipgram_model;
		this->src_te_model = src_te_model;
		this->tgt_te_model = tgt_te_model;
		this->src_word_vec = src_word_vec;
		this->tgt_word_vec = tgt_word_vec;
	}

	void clear()
	{
		src_dword_emb = MatrixXd::Zero(src_word_vec.vocb_size, word_dim);
		src_dW = MatrixXd::Zero(word_dim, src_word_vec.vocb_size);

		tgt_dword_emb = MatrixXd::Zero(tgt_word_vec.vocb_size, word_dim);
		tgt_dW = MatrixXd::Zero(word_dim, tgt_word_vec.vocb_size);
	}
};

#endif
