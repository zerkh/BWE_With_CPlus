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

static void* deepThread(void* arg)
{
	GCWEThread& gt = (GCWEThread&)arg;

	srand(time(0));
	for (int b = 0; b < gt.batch_size; b++)
	{
		int cur_sen = rand() / gt.sentences.size();

		vector<MatrixXd> derivations = trainOneSentence(*gt.gcwe_model, *gt.word_vec, gt.sentences[cur_sen], gt.window_size, gt.learning_rate);

		gt.dword_emb += derivations[0];
		gt.dW1 += derivations[1];
		gt.db1 += derivations[2];
		gt.dW2 += derivations[3];
		gt.dWg1 += derivations[4];
		gt.dbg1 += derivations[5];
		gt.dWg2 += derivations[6];
	}

	pthread_exit(NULL);
}

RowVectorXi getWindow(WordVec word_vec, string sentence, int window_size, int word_pos)
{
	vector<string> words = splitBySpace(sentence);
	vector<int> pos_of_word;

	for (int i = 0; i < window_size - 1; i++)
	{
		pos_of_word.push_back(0);
	}

	for (int w = 0; w < words.size(); w++)
	{
		pos_of_word.push_back(word_vec.m_word_id[words[w]]);
	}

	RowVectorXi window(window_size);
	for (int i = word_pos; i < window_size + word_pos; i++)
	{
		window(i - word_pos) = pos_of_word[i];
	}

	return window;
}

vector<MatrixXd> trainOneSentence(GCWE& gcwe_model, WordVec& word_vec, string sentence, int window_size, double learning_rate)
{

	//derivation items
	MatrixXd s_dword_emb = MatrixXd::Zero(word_vec.word_emb.rows(), word_vec.word_emb.cols());
	MatrixXd s_dW2 = MatrixXd::Zero(gcwe_model.W2.rows(), gcwe_model.W2.cols());
	MatrixXd s_dW1 = MatrixXd::Zero(gcwe_model.W1.rows(), gcwe_model.W1.cols());
	RowVectorXd s_db1 = RowVectorXd::Zero(gcwe_model.b1.cols());
	MatrixXd s_dWg2 = MatrixXd::Zero(gcwe_model.Wg2.rows(), gcwe_model.Wg2.cols());
	MatrixXd s_dWg1 = MatrixXd::Zero(gcwe_model.Wg1.rows(), gcwe_model.Wg1.cols());
	RowVectorXd s_dbg1 = RowVectorXd::Zero(gcwe_model.bg1.cols());


	vector<string> words = splitBySpace(sentence);

	vector<int> pos_of_word;

	for (int w = 0; w < words.size(); w++)
	{
		pos_of_word.push_back(word_vec.m_word_id[words[w]]);
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
		RowVectorXi x = getWindow(word_vec, sentence, window_size, w);

		vector<MatrixXd> derivations = gcwe_model.backward(word_vec, x, x_g);

		s_dword_emb += derivations[0];
		s_dW1 += derivations[1];
		s_db1 += derivations[2];
		s_dW2 += derivations[3];
		s_dWg1 += derivations[4];
		s_dbg1 += derivations[5];
		s_dWg2 += derivations[6];
	}

	s_dword_emb /= words.size();
	s_dW1 /= words.size();
	s_db1 /= words.size();
	s_dW2 /= words.size();
	s_dWg1 /= words.size();
	s_dbg1 /= words.size();
	s_dWg2 /= words.size();

	/*word_vec.word_emb += (s_dword_emb*learning_rate);
	gcwe_model.W1 += (s_dW1*learning_rate);
	gcwe_model.b1 += (s_db1*learning_rate);
	gcwe_model.W2 += (s_dW2*learning_rate);
	gcwe_model.Wg1 += (s_dWg1*learning_rate);
	gcwe_model.bg1 += (s_dbg1*learning_rate);
	gcwe_model.Wg2 += (s_dWg2*learning_rate);*/

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

void train(Config conf, GCWE& gcwe_model, WordVec& word_vec, string src_raw_file, double learning_rate, int epoch, int branch_size, int window_size)
{
	int thread_num = atoi(conf.get_para("thread_num").c_str());
	double start_clock, end_clock;
	pthread_t* pt = new pthread_t[thread_num];

	cout << "Init idf table..." << endl;
	start_clock = clock();
	word_vec.init_idf(src_raw_file);
	end_clock = clock();
	cout << "Complete to init idf table!" << "The cost of time is " << (end_clock-start_clock)/CLOCKS_PER_SEC << endl;

	ifstream src_raw_in(src_raw_file.c_str(), ios::in);
	if (!src_raw_in)
	{
		cout << "Cannot open " << src_raw_file << endl;
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
	GCWEThread* threadpara = new GCWEThread[thread_num];
	for (int i = 0; i < thread_num-1; i++)
	{
		threadpara[i].init(gcwe_model, word_vec, word_vec.word_dim, gcwe_model.hidden_dim, window_size, learning_rate);
		for (int i = ind; i < ind + sentence_per_thread; i++)
		{
			threadpara[i].sentences.push_back(sentences[i]);
		}
		ind += sentence_per_thread;
	}
	threadpara[thread_num - 1].init(gcwe_model, word_vec, word_vec.word_dim, gcwe_model.hidden_dim, window_size, learning_rate);
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
		MatrixXd s_dword_emb = MatrixXd::Zero(word_vec.word_emb.rows(), word_vec.word_emb.cols());
		MatrixXd s_dW2 = MatrixXd::Zero(gcwe_model.W2.rows(), gcwe_model.W2.cols());
		MatrixXd s_dW1 = MatrixXd::Zero(gcwe_model.W1.rows(), gcwe_model.W1.cols());
		RowVectorXd s_db1 = RowVectorXd::Zero(gcwe_model.b1.cols());
		MatrixXd s_dWg2 = MatrixXd::Zero(gcwe_model.Wg2.rows(), gcwe_model.Wg2.cols());
		MatrixXd s_dWg1 = MatrixXd::Zero(gcwe_model.Wg1.rows(), gcwe_model.Wg1.cols());
		RowVectorXd s_dbg1 = RowVectorXd::Zero(gcwe_model.bg1.cols());

		for (int t = 0; t < thread_num; t++)
		{
			s_dword_emb += threadpara[t].dword_emb;

			s_dW1 += threadpara[t].dW1;
			s_db1 += threadpara[t].db1;
			s_dW2 += threadpara[t].dW2;

			s_dWg1 += threadpara[t].dWg1;
			s_dbg1 += threadpara[t].dbg1;
			s_dWg2 += threadpara[t].dWg2;

			threadpara[t].clear();
		}

		word_vec.word_emb += (learning_rate*s_dword_emb / branch_size);

		gcwe_model.W1 += (learning_rate*s_dW1 / branch_size);
		gcwe_model.b1 += (learning_rate*s_db1 / branch_size);
		gcwe_model.W2 += (learning_rate*s_dW2 / branch_size);
		gcwe_model.Wg1 += (learning_rate*s_dWg1 / branch_size);
		gcwe_model.bg1 += (learning_rate*s_dbg1 / branch_size);
		gcwe_model.Wg2 += (learning_rate*s_dWg2 / branch_size);

		cout << "Epoch " << e << " complete!" << endl;
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
	string output_dir = conf.get_para("output_dir");
	int word_dim = atoi(conf.get_para("word_dim").c_str());
	int hidden_dim = atoi(conf.get_para("hidden_dim").c_str());
	int window_size = atoi(conf.get_para("window_size").c_str());
	int neg_sample = atoi(conf.get_para("neg_sample").c_str());
	double learning_rate = atof(conf.get_para("learning_rate").c_str());
	int epoch = atoi(conf.get_para("epoch").c_str());
	int branch_size = atoi(conf.get_para("branch_size").c_str());

	//init any
	double start_clock, end_clock;

	WordVec src_word_vec(word_dim, src_vocab_file);

	GCWE gcwe_model(word_dim, hidden_dim, window_size, neg_sample);

	string src_raw_file = conf.get_para("src_raw_file");

	//training
	cout << "Start training......" << endl;
	start_clock = clock();
	train(conf, gcwe_model, src_word_vec, src_raw_file, learning_rate, epoch, branch_size, window_size);
	end_clock = clock();
	cout << "Complete to train word vectors! The cost of time is " << (end_clock - start_clock) / CLOCKS_PER_SEC << endl;

	cout << "Start saving word vector......" << endl;
	start_clock = clock();
	src_word_vec.saveWordVec(output_dir);
	end_clock = clock();
	cout << "Complete to save word vectors! The cost of time is " << (end_clock - start_clock) / CLOCKS_PER_SEC << endl;

	return 0;
}
