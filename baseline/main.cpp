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

//                       _oo0oo_
//                      o8888888o
//                      88" . "88
//                      (| -_- |)
//                      0\  =  /0
//                    ___/`---'\___
//                  .' \\|     |// '.
//                 / \\|||  :  |||// \
//                / _||||| -:- |||||- \
//               |   | \\\  -  /// |   |
//               | \_|  ''\---/''  |_/ |
//               \  .-\__  '-'  ___/-. /
//             ___'. .'  /--.--\  `. .'___
//          ."" '<  `.___\_<|>_/___.' >' "".
//         | | :  `- \`.;`\ _ /`;.`/ - ` : | |
//         \  \ `_.   \_ __\ /__ _/   .-` /  /
//     =====`-.____`.___ \_____/___.-`___.-'=====
//                       `=---='
//
//
//     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//               ���汣��         ����BUG
//
//


/*
train a pair of sentence
return d_word_emb, d_W
*/
vector<MatrixXd> trainOneSentence(SkipGram& skipgram_model, TE& te_model, WordVec src_word_vec, WordVec& tgt_word_vec, string sentence, int window_size, double learning_rate, double lambda)
{
	//derivation items
	MatrixXd s_dword_emb = MatrixXd::Zero(tgt_word_vec.word_emb.rows(), tgt_word_vec.word_emb.cols());
	MatrixXd s_dW = MatrixXd::Zero(skipgram_model.W.rows(), skipgram_model.W.cols());
	double start_clock, end_clock;

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

		start_clock = clock();
		vector<MatrixXd> derivations = skipgram_model.backward(tgt_word_vec, x, c);
		end_clock = clock();
		//cout << "The time of do a skip-gram backward is " << (end_clock-start_clock)/CLOCKS_PER_SEC << endl;

		s_dword_emb += derivations[0];
		s_dW += derivations[1];
	}

	
	start_clock = clock();
	vector<MatrixXd> derivations = te_model.backward(src_word_vec, tgt_word_vec);
	end_clock = clock();
	//cout << "The time of do a te backward is " << (end_clock-start_clock)/CLOCKS_PER_SEC << endl;

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
	double start_clock, end_clock;

	srand(time(0));
	for (int b = 0; b < gt->batch_size; b++)
	{
		int cur_src_sen = rand() % gt->src_sentences.size();
		int cur_tgt_sen = rand() % gt->tgt_sentences.size();

		start_clock = clock();
		vector<MatrixXd> derivations = trainOneSentence(gt->src_skipgram_model, gt->src_te_model, 
														gt->tgt_word_vec, gt->src_word_vec, gt->src_sentences[cur_src_sen], 
														gt->window_size, gt->learning_rate, gt->lambda);
		end_clock = clock();
		cout << "The time of train a source sentence is " << (end_clock-start_clock)/CLOCKS_PER_SEC << endl;

		start_clock = clock();
		gt->src_dword_emb += derivations[0];
		gt->src_dW += derivations[1];
		gt->src_word_count += splitBySpace(gt->src_sentences[cur_src_sen]).size();
		end_clock = clock();
		//cout << "The time of update is " << (end_clock-start_clock)/CLOCKS_PER_SEC << endl;

		derivations.clear();

		start_clock = clock();
		derivations = trainOneSentence(gt->tgt_skipgram_model, gt->tgt_te_model,
			gt->src_word_vec, gt->tgt_word_vec, gt->tgt_sentences[cur_tgt_sen],
			gt->window_size, gt->learning_rate, gt->lambda);
		end_clock = clock();
		cout << "The time of train a target sentence is " << (end_clock - start_clock) / CLOCKS_PER_SEC << endl;

		gt->tgt_dword_emb += derivations[0];
		gt->tgt_dW += derivations[1];
		gt->tgt_word_count += splitBySpace(gt->tgt_sentences[cur_tgt_sen]).size();
	}

	pthread_exit(NULL);
}

void trainWordVec(Config conf, SkipGram& src_skipgram_model, SkipGram& tgt_skipgram_model,
					TE& src_te_model, TE& tgt_te_model,
					WordVec& src_word_vec, WordVec& tgt_word_vec,  
					string src_raw_file, string tgt_raw_file, 
					double learning_rate, int epoch, int branch_size, int window_size)
{
	int thread_num = atoi(conf.get_para("thread_num").c_str());
	double lambda = atof(conf.get_para("lambda").c_str());
	double start_clock, end_clock;
	pthread_t* pt = new pthread_t[thread_num];

	ifstream src_raw_in(src_raw_file.c_str(), ios::in);
	if (!src_raw_in)
	{
		cout << "Cannot open " << src_raw_file << endl;
		exit(0);
	}

	ifstream tgt_raw_in(tgt_raw_file.c_str(), ios::in);
	if (!tgt_raw_in)
	{
		cout << "Cannot open " << tgt_raw_file << endl;
		exit(0);
	}

	string sentence;
	vector<string> src_sentences;
	vector<string> tgt_sentences;

	while (getline(src_raw_in, sentence))
	{
		src_sentences.push_back(sentence);
	}

	while (getline(tgt_raw_in, sentence))
	{
		tgt_sentences.push_back(sentence);
	}

	src_raw_in.close();
	tgt_raw_in.close();

	//init multi-thread information
	int sentence_branch = branch_size / thread_num;
	int src_sentence_per_thread = src_sentences.size() / thread_num;
	int tgt_sentence_per_thread = tgt_sentences.size() / thread_num;
	int src_ind = 0;
	int tgt_ind = 0;

	TEThread* threadpara = new TEThread[thread_num];
	for (int t = 0; t < thread_num - 1; t++)
	{
		threadpara[t].init(src_skipgram_model, tgt_skipgram_model,
							src_te_model, tgt_te_model,
							src_word_vec, tgt_word_vec,
							tgt_word_vec.word_dim, window_size, learning_rate, lambda);
		for (int i = src_ind; i < src_ind + src_sentence_per_thread; i++)
		{
			threadpara[t].src_sentences.push_back(src_sentences[i]);
		}

		for (int i = tgt_ind; i < tgt_ind + tgt_sentence_per_thread; i++)
		{
			threadpara[t].tgt_sentences.push_back(tgt_sentences[i]);
		}

		threadpara[t].batch_size = sentence_branch;
		src_ind += src_sentence_per_thread;
		tgt_ind += tgt_sentence_per_thread;
	}
	
	threadpara[thread_num - 1].init(src_skipgram_model, tgt_skipgram_model, 
									src_te_model, tgt_te_model,
									src_word_vec, tgt_word_vec,
									tgt_word_vec.word_dim, window_size, learning_rate, lambda);
	threadpara[thread_num - 1].batch_size = branch_size - (thread_num-1)*sentence_branch;
	for (int i = src_ind; i < src_sentences.size(); i++)
	{
		threadpara[thread_num - 1].src_sentences.push_back(src_sentences[i]);
	}

	for (int i = tgt_ind; i < tgt_sentences.size(); i++)
	{
		threadpara[thread_num - 1].tgt_sentences.push_back(tgt_sentences[i]);
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
		MatrixXd s_src_dword_emb = MatrixXd::Zero(src_word_vec.word_emb.rows(), src_word_vec.word_emb.cols());
		MatrixXd s_src_dW = MatrixXd::Zero(src_skipgram_model.W.rows(), src_skipgram_model.W.cols());
		MatrixXd s_tgt_dword_emb = MatrixXd::Zero(tgt_word_vec.word_emb.rows(), tgt_word_vec.word_emb.cols());
		MatrixXd s_tgt_dW = MatrixXd::Zero(tgt_skipgram_model.W.rows(), tgt_skipgram_model.W.cols());
		int src_word_count = 0;
		int tgt_word_count = 0;

		for (int t = 0; t < thread_num; t++)
		{
			s_src_dword_emb += threadpara[t].src_dword_emb;
			s_src_dW += threadpara[t].src_dW;
			src_word_count += threadpara[t].src_word_count;

			s_tgt_dword_emb += threadpara[t].tgt_dword_emb;
			s_tgt_dW += threadpara[t].tgt_dW;
			tgt_word_count += threadpara[t].tgt_word_count;

			cout << "score: " << threadpara[t].evaluate() << endl;

			threadpara[t].clear();
		}

		src_word_vec.word_emb += (learning_rate*s_src_dword_emb / src_word_count);
		tgt_word_vec.word_emb += (learning_rate*s_tgt_dword_emb / tgt_word_count);

		src_skipgram_model.W += (learning_rate*s_src_dW / src_word_count);
		tgt_skipgram_model.W += (learning_rate*s_tgt_dW / tgt_word_count);

		cout << "Epoch " << e + 1 << " complete!" << endl;

		for (int t = 0; t < thread_num; t++)
		{
			threadpara[t].update(src_skipgram_model, tgt_skipgram_model,
									src_te_model, tgt_te_model,
									src_word_vec, tgt_word_vec);
		}
	}

	src_raw_in.close();
	delete[] threadpara;
	delete pt;
}

int main(int argc, char* argv[])
{
	if (argc != 2)
	{
		cerr << "Usage: ./te Config.conf" << endl;
	}

	Config conf(argv[1]);

	//get config
	string src_vocab_file = conf.get_para("src_vocab_file");
	string tgt_vocab_file = conf.get_para("tgt_vocab_file");
	int word_dim = atoi(conf.get_para("word_dim").c_str());
	string output_dir = conf.get_para("output_dir");
	int window_size = atoi(conf.get_para("window_size").c_str());
	double learning_rate = atof(conf.get_para("learning_rate").c_str());
	int epoch = atoi(conf.get_para("epoch").c_str());
	int branch_size = atoi(conf.get_para("branch_size").c_str());
	string src_align_table_file = conf.get_para("src_align_table");
	string tgt_align_table_file = conf.get_para("tgt_align_table");
	string tgt_raw_file = conf.get_para("tgt_raw_file");
	string src_raw_file = conf.get_para("src_raw_file");
	string tgt_skipgram_file = conf.get_para("tgt_skipgram_file");
	string src_skipgram_file = conf.get_para("src_skipgram_file");

	double start_clock, end_clock;

	//init word vectors
	WordVec src_word_vec(word_dim, src_vocab_file);
	WordVec tgt_word_vec(word_dim, tgt_vocab_file);

	//load word vector of source language pre-trained by GCWE
	cout << "Load src word vectors from \"" << conf.get_para("src_word_vec") << "\"......" << endl;
	src_word_vec.loadWordVec(conf.get_para("src_word_vec"));
	tgt_word_vec.loadWordVec(conf.get_para("tgt_word_vec"));

	//init tgt skip-gram model and translation-equivalence
	cout << "Init tgt skip-gram model and TE model ......" << endl;
	SkipGram src_skipgram_model(src_word_vec.vocb_size, word_dim);
	SkipGram tgt_skipgram_model(tgt_word_vec.vocb_size, word_dim);
	TE src_te_model(tgt_word_vec, src_word_vec);
	TE tgt_te_model(src_word_vec, tgt_word_vec);

	//load source skip-gram model
	src_skipgram_model.loadModel(src_skipgram_file);

	//init target word vector with equivalence and source word vector
	start_clock = clock();
	cout << "Reading alignment table......" << endl;
	src_te_model.readAlignTable(src_align_table_file, tgt_word_vec, src_word_vec);
	tgt_te_model.readAlignTable(tgt_align_table_file, src_word_vec, tgt_word_vec);
	end_clock = clock();
	cout << "Complete to read alignment table! The cost of time is " << (end_clock-start_clock) / CLOCKS_PER_SEC << endl;

	//training
	cout << "Start training......" << endl;
	start_clock = clock();
	trainWordVec(conf, src_skipgram_model, tgt_skipgram_model,
					src_te_model, tgt_te_model,
					src_word_vec, tgt_word_vec,
					src_raw_file, tgt_raw_file,
					learning_rate, epoch, branch_size, window_size);
	end_clock = clock();
	cout << "Complete to train word vectors! The cost of time is " << (end_clock - start_clock) / CLOCKS_PER_SEC << endl;

	//save
	cout << "Start saving word vector......" << endl;
	start_clock = clock();
	src_word_vec.saveWordVec(output_dir, "src");
	tgt_word_vec.saveWordVec(output_dir, "tgt");
	end_clock = clock();
	cout << "Complete to save word vectors! The cost of time is " << (end_clock - start_clock) / CLOCKS_PER_SEC << endl;

	cout << "Start saving skip-gram model......" << endl;
	start_clock = clock();
	src_skipgram_model.saveModel(src_skipgram_file);
	tgt_skipgram_model.saveModel(tgt_skipgram_file);
	end_clock = clock();
	cout << "Complete to save skip-gram model! The cost of time is " << (end_clock - start_clock) / CLOCKS_PER_SEC << endl;

	return 0;
}
