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
	int src_word_count;
	int tgt_word_count;
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

		this->src_word_count = 0;
		this->tgt_word_count = 0;
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

		src_word_count = 0;
		tgt_word_count = 0;
	}
};

class TEThread
{
public:
	MatrixXd src_dword_emb;
	MatrixXd src_dW;
	MatrixXd tgt_dword_emb;
	MatrixXd tgt_dW;

	int src_word_count;
	int tgt_word_count;
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

		this->src_word_count = 0;
		this->tgt_word_count = 0;
		this->lambda = lambda;
		this->learning_rate = learning_rate;
		this->word_dim = word_dim;
		this->window_size = window_size;
	}

	double evaluate()
	{
		double f_score = 0;

		f_score += src_te_model.forward(tgt_word_vec, src_word_vec);
		f_score += tgt_te_model.forward(src_word_vec, tgt_word_vec);

		vector<string> src_words = splitBySpace(src_sentences[0]);
		vector<string> tgt_words = splitBySpace(tgt_sentences[0]);

		vector<int> pos_of_word;

		for (int w = 0; w < tgt_words.size(); w++)
		{
			pos_of_word.push_back(tgt_word_vec.m_word_id[tgt_words[w]]);
		}

		//get global context
		RowVectorXi x_g(tgt_words.size());
		for (int i = 0; i < tgt_words.size(); i++)
		{
			x_g(i) = pos_of_word[i];
		}

		double tmp_score = 0;

		//train one sentence
		for (int w = 0; w < tgt_words.size(); w++)
		{
			RowVectorXi c = getWindow(tgt_word_vec, tgt_sentences[0], window_size, w);

			int x = c(window_size - 1);
			c = c.head(window_size - 1);

			tmp_score += tgt_skipgram_model.forward(tgt_word_vec, x, c);
		}

		tmp_score /= tgt_words.size();
		f_score += tmp_score;

		pos_of_word.clear();

		for (int w = 0; w < src_words.size(); w++)
		{
			pos_of_word.push_back(src_word_vec.m_word_id[src_words[w]]);
		}

		//get global context
		x_g = RowVectorXi(src_words.size());
		for (int i = 0; i < src_words.size(); i++)
		{
			x_g(i) = pos_of_word[i];
		}

		tmp_score = 0;

		//train one sentence
		for (int w = 0; w < src_words.size(); w++)
		{
			RowVectorXi c = getWindow(src_word_vec, src_sentences[0], window_size, w);

			int x = c(window_size - 1);
			c = c.head(window_size - 1);

			tmp_score += src_skipgram_model.forward(src_word_vec, x, c);
		}

		tmp_score /= src_words.size();
		f_score += tmp_score;

		return f_score;
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

		src_word_count = 0;
		tgt_word_count = 0;
	}
};

#endif
