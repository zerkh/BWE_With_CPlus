#ifndef __SKIPGRAM__
#define __SKIPGRAM__

#include "Utils.h"
#include "WordVec.h"
#include "limits"

class SkipGram
{
public:
	int vocb_size;
	int word_dim;

	MatrixXd W;

	SkipGram() {};

	SkipGram(int vocb_size, int word_dim);

	double forward(WordVec& word_vec, int x, RowVectorXi c);

	vector<MatrixXd> backward(WordVec& word_vec, int x, RowVectorXi c);

	void saveModel(string save_file);

	void loadModel(string load_file);
};

void train(Config conf, SkipGram& skipgram_model, WordVec& word_vec, string src_raw_file, double learning_rate, int epoch, int branch_size, int window_size);

vector<MatrixXd> trainOneSentence(SkipGram& skipgram_model, WordVec& word_vec, string sentence, int window_size, double learning_rate);

RowVectorXi getWindow(WordVec word_vec, string sentence, int window_size, int word_pos);

#endif
