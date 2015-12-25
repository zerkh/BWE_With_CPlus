#include "LSOP.h"

LSOP::LSOP(SparseMatrix<double>& alignTable)
{
	this->alignTable = alignTable;
}

double LSOP::forward(int word_pos, WordVec src_word_vec, WordVec tgt_word_vec)
{
	double sum = 0;
	for (int col = 0; col < alignTable.cols(); col++)
	{
		if (alignTable.coeffRef(word_pos, col) != 0)
		{
			sum += exp(src_word_vec.word_emb.row(word_pos)*tgt_word_vec.word_emb.row(col).transpose());
		}
	}

	double result = 0;

	for (int col = 0; col < alignTable.cols(); col++)
	{
		if (alignTable.coeffRef(word_pos, col) != 0)
		{
			double tmp = exp(src_word_vec.word_emb.row(word_pos)*tgt_word_vec.word_emb.row(col).transpose());
			result -= alignTable.coeffRef(word_pos, col)*log(tmp / sum);
		}
	}

	return result;
}

vector<MatrixXd> LSOP::backward(int word_pos, WordVec src_word_vec, WordVec tgt_word_vec)
{
	double sum = 0;
	VectorXd component = VectorXd::Zero(src_word_vec.word_dim);
	for (int col = 0; col < alignTable.cols(); col++)
	{
		if (alignTable.coeffRef(word_pos, col) != 0)
		{
			sum += exp(src_word_vec.word_emb.row(word_pos)*tgt_word_vec.word_emb.row(col).transpose());
			component += tgt_word_vec.word_emb.row(col)*exp(src_word_vec.word_emb.row(word_pos)*tgt_word_vec.word_emb.row(col).transpose());
		}
	}

	component /= sum;
	MatrixXd dword_vec = MatrixXd::Zero(src_word_vec.vocb_size, src_word_vec.word_dim);

	for (int col = 0; col < alignTable.cols(); col++)
	{
		if (alignTable.coeffRef(word_pos, col) != 0)
		{
			VectorXd tmp = VectorXd::Zero(src_word_vec.word_dim);
			tmp = src_word_vec.word_emb.row(col) - component;
			tmp *= alignTable.coeffRef(word_pos, col);
			dword_vec.row(word_pos) += tmp;
		}
	}

	vector<MatrixXd> derivation;
	derivation.push_back(dword_vec);

	return derivation;
}