#include "SkipGram.h"
#include "ThreadPara.h"

SkipGram::SkipGram(int vocb_size, int word_dim)
{
	this->vocb_size = vocb_size;
	this->word_dim = word_dim;

	W = MatrixXd::Random(word_dim, vocb_size);
}

double SkipGram::forward(WordVec& word_vec, int x, RowVectorXi c)
{
	RowVectorXd U = word_vec.word_emb.row(x) * W;

	vector<double> probs;

	for (int col = 0; col < U.cols(); col++)
	{
		U(col) = exp(U(col));
	}

	double sum = U.sum();

	for (int i = 0; i < c.cols(); i++)
	{
		probs.push_back(U(c(i))/sum);
	}

	double score = 1;

	for (int i = 0; i < probs.size(); i++)
	{
		score *= probs[i];
	}

	score = -1.0 * log(score);

	return score;
}

vector<MatrixXd> SkipGram::backward(WordVec& word_vec, int x, RowVectorXi c)
{
	RowVectorXd U = word_vec.word_emb.row(x) * W;

	vector<double> probs;

	for (int col = 0; col < U.cols(); col++)
	{
		U(col) = exp(U(col));
	}

	double sum = U.sum();

	for (int i = 0; i < c.cols(); i++)
	{
		probs.push_back(U(c(i)) / sum);
	}

	MatrixXd dW = MatrixXd::Zero(word_dim, vocb_size);
	MatrixXd dword_emb = MatrixXd::Zero(vocb_size, word_dim);

	for (int i = 0; i < c.size(); i++)
	{
		dW.col(c(i)) += (word_vec.word_emb.row(x).transpose() * (probs[i] - 1));
		dword_emb.row(x) += (W.col(c(i)).transpose() * (probs[i] - 1));
	}

	vector<MatrixXd> derivation;
	derivation.push_back(dword_emb);
	derivation.push_back(dW);

	return derivation;
}

void SkipGram::saveModel(string save_file)
{
	ofstream out(save_file.c_str(), ios::out);

	out << "W:" << endl;
	out << W << endl;
}

/***********************/
/*About model training */
/***********************/
static void* deepThread(void* arg)
{
	SkipGramThread* gt = (SkipGramThread*)arg;

	srand(time(0));
	for (int b = 0; b < gt->batch_size; b++)
	{
		int cur_sen = rand() % gt->sentences.size();

		vector<MatrixXd> derivations = trainOneSentence(gt->skipgram_model, gt->word_vec, gt->sentences[cur_sen], gt->window_size, gt->learning_rate);
		gt->dword_emb += derivations[0];
		gt->dW += derivations[1];
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

vector<MatrixXd> trainOneSentence(SkipGram& skipgram_model, WordVec& word_vec, string sentence, int window_size, double learning_rate)
{
	//derivation items
	MatrixXd s_dword_emb = MatrixXd::Zero(word_vec.word_emb.rows(), word_vec.word_emb.cols());
	MatrixXd s_dW = MatrixXd::Zero(skipgram_model.W.rows(), skipgram_model.W.cols());

	vector<string> words = splitBySpace(sentence);

	vector<int> pos_of_word;

	for (int w = 0; w < words.size(); w++)
	{
		pos_of_word.push_back(word_vec.m_word_id[words[w]]);
	}

	//train one sentence
	for (int w = 0; w < words.size(); w++)
	{
		RowVectorXi c = getWindow(word_vec, sentence, window_size, w);

		int x = c(window_size - 1);
		c = c.head(window_size - 1);

		vector<MatrixXd> derivations = skipgram_model.backward(word_vec, x, c);

		s_dword_emb += derivations[0];
		s_dW += derivations[1];
	}

	vector<MatrixXd> derivations;
	derivations.push_back(s_dword_emb);
	derivations.push_back(s_dW);

	return derivations;
}

void train(Config conf, SkipGram& skipgram_model, WordVec& word_vec, string src_raw_file, double learning_rate, int epoch, int branch_size, int window_size)
{
	int thread_num = atoi(conf.get_para("thread_num").c_str());
	double start_clock, end_clock;
	pthread_t* pt = new pthread_t[thread_num];

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
	SkipGramThread* threadpara = new SkipGramThread[thread_num];
	for (int t = 0; t < thread_num - 1; t++)
	{
		threadpara[t].init(skipgram_model, word_vec, word_vec.word_dim, window_size, learning_rate);
		for (int i = ind; i < ind + sentence_per_thread; i++)
		{
			threadpara[t].sentences.push_back(sentences[i]);
		}
		ind += sentence_per_thread;
		threadpara[t].batch_size = sentence_branch;
	}
	threadpara[thread_num - 1].init(skipgram_model, word_vec, word_vec.word_dim, window_size, learning_rate);
	threadpara[thread_num - 1].batch_size = branch_size - (thread_num - 1)*sentence_branch;
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
		MatrixXd s_dW = MatrixXd::Zero(skipgram_model.W.rows(), skipgram_model.W.cols());

		for (int t = 0; t < thread_num; t++)
		{
			s_dword_emb += threadpara[t].dword_emb;

			s_dW += threadpara[t].dW;

			threadpara[t].clear();
		}

		word_vec.word_emb += (learning_rate*s_dword_emb / branch_size);

		skipgram_model.W += (learning_rate*s_dW / branch_size);

		cout << "Epoch " << e + 1 << " complete!" << endl;

		for (int t = 0; t < thread_num; t++)
		{
			threadpara[t].update(skipgram_model, word_vec);
		}
	}

	src_raw_in.close();
	delete[]threadpara;
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
	string src_gcwe_file = conf.get_para("src_gcwe_file");

	//init any
	double start_clock, end_clock;

	WordVec src_word_vec(word_dim, src_vocab_file);

	SkipGram skipgram_model(src_word_vec.vocb_size, word_dim);

	string src_raw_file = conf.get_para("src_raw_file");

	//training
	cout << "Start training......" << endl;
	start_clock = clock();
	train(conf, skipgram_model, src_word_vec, src_raw_file, learning_rate, epoch, branch_size, window_size);
	end_clock = clock();
	cout << "Complete to train word vectors! The cost of time is " << (end_clock - start_clock) / CLOCKS_PER_SEC << endl;

	cout << "Start saving word vector......" << endl;
	start_clock = clock();
	src_word_vec.saveWordVec(output_dir);
	end_clock = clock();
	cout << "Complete to save word vectors! The cost of time is " << (end_clock - start_clock) / CLOCKS_PER_SEC << endl;

	cout << "Start saving skip-gram model......" << endl;
	start_clock = clock();
	skipgram_model.saveModel(src_gcwe_file);
	end_clock = clock();
	cout << "Complete to save skip-gram model! The cost of time is " << (end_clock - start_clock) / CLOCKS_PER_SEC << endl;

	return 0;
}
