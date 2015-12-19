#ifndef __THREADPARA__
#define __THREADPARA__

#include "Utils.h"
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

#endif
