#include "Eigen/Dense"
#include <iostream>
#include "Config.h"
#include "Utils.h"
#include "SkipGram.h"
#include "TE.h"
#include "WordVec.h"
#include "ThreadPara.h"
using namespace std;
using namespace Eigen;

/*
Baseline of our work
Reference Zou EMNLP 2013
Author kh
*/
vector<MatrixXd> trainOneSentence(SkipGram& skipgram_model, TE& te_model, WordVec src_word_vec, WordVec& tgt_word_vec, string sentence, int window_size, double learning_rate, double lambda)
{

	//derivation items
	MatrixXd s_dword_emb = MatrixXd::Zero(tgt_word_vec.word_emb.rows(), tgt_word_vec.word_emb.cols());
	MatrixXd s_dW = MatrixXd::Zero(skipgram_model.W.rows(), skipgram_model.W.cols());


	vector<string> words = splitBySpace(sentence);

	vector<int> pos_of_word;

	for (int w = 0; w < words.size(); w++)
	{
		pos_of_word.push_back(tgt_word_vec.m_word_id[words[w]]);
	}

	//get global context
	RowVectorXi x_g(words.size());
	for (int i = 0; i < words.size(); i++)
	{
		x_g(i) = pos_of_word[i];
	}

	//train one sentence
	for (int w = 0; w < words.size(); w++)
	{
		RowVectorXi c = getWindow(tgt_word_vec, sentence, window_size, w);

		int x = c(window_size - 1);
		c = c.head(window_size - 1);

		vector<MatrixXd> derivations = skipgram_model.backward(tgt_word_vec, x, c);

		s_dword_emb += derivations[0];
		s_dW += derivations[1];
	}

	vector<MatrixXd> derivations = te_model.backward(src_word_vec, tgt_word_vec);
	MatrixXd te_dword_emb = MatrixXd::Zero(s_dword_emb.rows(), s_dword_emb.cols());
	for (int w = 0; w < words.size(); w++)
	{
		te_dword_emb.row(tgt_word_vec.m_word_id[words[w]]) = derivations[0].row(tgt_word_vec.m_word_id[words[w]]);
	}

	s_dword_emb += (lambda*te_dword_emb);

	derivations.clear();
	derivations.push_back(s_dword_emb);
	derivations.push_back(s_dW);

	return derivations;
}

static void* deepThread(void* arg)
{
	TEThread* gt = (TEThread*)arg;

	srand(time(0));
	for (int b = 0; b < gt->batch_size; b++)
	{
		int cur_sen = rand() / gt->sentences.size();

		vector<MatrixXd> derivations = trainOneSentence(gt->skipgram_model, gt->te_model, gt->src_word_vec, gt->tgt_word_vec, gt->sentences[cur_sen], gt->window_size, gt->learning_rate, gt->lambda);

		gt->dword_emb += derivations[0];
		gt->dW += derivations[1];
	}

	pthread_exit(NULL);
}

void trainTgtWordVec(Config conf, SkipGram& skipgram_model, TE& te_model, WordVec src_word_vec, WordVec& tgt_word_vec, string tgt_raw_file, double learning_rate, int epoch, int branch_size, int window_size)
{
	int thread_num = atoi(conf.get_para("thread_num").c_str());
	double lambda = atof(conf.get_para("lambda").c_str());
	double start_clock, end_clock;
	pthread_t* pt = new pthread_t[thread_num];

	cout << "Init idf table..." << endl;
	start_clock = clock();
	tgt_word_vec.init_idf(tgt_raw_file);
	end_clock = clock();
	cout << "Complete to init idf table!" << "The cost of time is " << (end_clock - start_clock) / CLOCKS_PER_SEC << endl;

	ifstream src_raw_in(tgt_raw_file.c_str(), ios::in);
	if (!src_raw_in)
	{
		cout << "Cannot open " << tgt_raw_file << endl;
		exit(0);
	}


	string sentence;
	vector<string> sentences;

	while (getline(src_raw_in, sentence))
	{
		sentences.push_back(sentence);
	}

	//init multi-thread information
	int sentence_branch = branch_size / thread_num;
	int sentence_per_thread = sentences.size() / thread_num;
	int ind = 0;
	TEThread* threadpara = new TEThread[thread_num];
	for (int i = 0; i < thread_num - 1; i++)
	{
		threadpara[i].init(skipgram_model, te_model, src_word_vec, tgt_word_vec, tgt_word_vec.word_dim, window_size, learning_rate, lambda);
		for (int i = ind; i < ind + sentence_per_thread; i++)
		{
			threadpara[i].sentences.push_back(sentences[i]);
		}
		ind += sentence_per_thread;
	}
	threadpara[thread_num - 1].init(skipgram_model, te_model, src_word_vec, tgt_word_vec, tgt_word_vec.word_dim, window_size, learning_rate, lambda);
	threadpara[thread_num - 1].batch_size = branch_size - thread_num*sentence_branch;
	for (int i = ind; i < sentences.size(); i++)
	{
		threadpara[thread_num - 1].sentences.push_back(sentences[i]);
	}

	for (int e = 0; e < epoch; e++)
	{
		cout << "Start training epoch " << e + 1 << endl;
		for (int t = 0; t < thread_num; t++)
		{
			pthread_create(&pt[t], NULL, deepThread, (void *)(threadpara + t));
		}
		for (int t = 0; t < thread_num; t++)
		{
			pthread_join(pt[t], NULL);
		}

		//update
		MatrixXd s_dword_emb = MatrixXd::Zero(tgt_word_vec.word_emb.rows(), tgt_word_vec.word_emb.cols());
		MatrixXd s_dW = MatrixXd::Zero(skipgram_model.W.rows(), skipgram_model.W.cols());

		for (int t = 0; t < thread_num; t++)
		{
			s_dword_emb += threadpara[t].dword_emb;

			s_dW += threadpara[t].dW;

			threadpara[t].clear();
		}

		tgt_word_vec.word_emb += (learning_rate*s_dword_emb / branch_size);

		skipgram_model.W += (learning_rate*s_dW / branch_size);

		cout << "Epoch " << e + 1 << " complete!" << endl;

		for (int t = 0; t < thread_num; t++)
		{
			threadpara[t].update(skipgram_model, te_model, src_word_vec, tgt_word_vec);
		}
	}

	src_raw_in.close();
	delete threadpara;
	delete pt;
}

int main()
{
	Config conf("Config.conf");

	//get config
	string src_vocab_file = conf.get_para("src_vocab_file");
	string tgt_vocab_file = conf.get_para("tgt_vocab_file");
	int word_dim = atoi(conf.get_para("word_dim").c_str());
	string output_dir = conf.get_para("output_dir");
	int hidden_dim = atoi(conf.get_para("hidden_dim").c_str());
	int window_size = atoi(conf.get_para("window_size").c_str());
	double learning_rate = atof(conf.get_para("learning_rate").c_str());
	int epoch = atoi(conf.get_para("epoch").c_str());
	int branch_size = atoi(conf.get_para("branch_size").c_str());
	string align_table_file = conf.get_para("align_table");
	string tgt_raw_file = conf.get_para("tgt_raw_file");
	string tgt_gcwe_file = conf.get_para("tgt_gcwe_file");

	double start_clock, end_clock;

	//init word vectors
	WordVec src_word_vec(word_dim, src_vocab_file);
	WordVec tgt_word_vec(word_dim, tgt_vocab_file);

	//load word vector of source language pre-trained by GCWE
	cout << "Load src word vectors from \"" << conf.get_para("src_word_vec") << "\"......" << endl;
	src_word_vec.loadWordVec(conf.get_para("src_word_vec"));

	//init tgt GCWE model and translation-equivalence
	cout << "Init tgt GCWE model and TE model ......" << endl;
	SkipGram skipgram_model(tgt_word_vec.vocb_size, word_dim);
	TE te_model(src_word_vec, tgt_word_vec);

	//init target word vector with equivalence and source word vector
	cout << "Reading alignment table......" << endl;
	te_model.readAlignTable(align_table_file, src_word_vec, tgt_word_vec);

	te_model.initTgtWordVec(src_word_vec, tgt_word_vec);

	//training
	cout << "Start training......" << endl;
	start_clock = clock();
	trainTgtWordVec(conf, skipgram_model, te_model, src_word_vec, tgt_word_vec, tgt_raw_file, learning_rate, epoch, branch_size, window_size);
	end_clock = clock();
	cout << "Complete to train word vectors! The cost of time is " << (end_clock - start_clock) / CLOCKS_PER_SEC << endl;

	//save
	cout << "Start saving word vector......" << endl;
	start_clock = clock();
	src_word_vec.saveWordVec(output_dir);
	end_clock = clock();
	cout << "Complete to save word vectors! The cost of time is " << (end_clock - start_clock) / CLOCKS_PER_SEC << endl;

	cout << "Start saving tgt GCWE model......" << endl;
	start_clock = clock();
	skipgram_model.saveModel(tgt_gcwe_file);
	end_clock = clock();
	cout << "Complete to save tgt GCWE model! The cost of time is " << (end_clock - start_clock) / CLOCKS_PER_SEC << endl;

	return 0;
}
