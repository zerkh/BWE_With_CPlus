#include "SkipGram.h"

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
	derivation.push_back(dW);
	derivation.push_back(dword_emb);

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