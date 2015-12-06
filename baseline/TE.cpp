#include "TE.h"

TE::TE(WordVec src_word_vec, WordVec tgt_word_vec)
{
	src_vocb_size = src_word_vec.vocb_size;
	tgt_vocb_size = tgt_word_vec.vocb_size;

	alignTable = MatrixXd::Zero(tgt_vocb_size, src_vocb_size);
}

/*
input: 
filename: example: {alignTable.xml} format: {src_word tgt_word align_times}
*/
void TE::readAlignTable(string filename, WordVec src_word_vec, WordVec tgt_word_vec)
{
	ifstream in(filename.c_str(), ios::in);

	string line;
	while (getline(in, line))
	{
		vector<string> components = splitBySpace(line);

		alignTable(tgt_word_vec.m_word_id[components[1]], src_word_vec.m_word_id[components[0]]) = atoi(components[2].c_str())+1;
	}

	for (int row = 0; row < alignTable.rows(); row++)
	{
		alignTable.row(row) /= alignTable.row(row).sum();
	}
}

void TE::initTgtWordVec(WordVec src_word_vec, WordVec& tgt_word_vec)
{
	tgt_word_vec.word_emb.fill(0.0);

	for (int t = 0; t < tgt_vocb_size; t++)
	{
		for (int s = 0; s < src_vocb_size; s++)
		{
			tgt_word_vec.word_emb.row(t) += alignTable(t,s)*src_word_vec.word_emb.row(s);
		}
	}
}

double TE::forward(WordVec src_word_vec, WordVec tgt_word_vec)
{
	MatrixXd error_mat = tgt_word_vec.word_emb - alignTable*src_word_vec.word_emb;

	double s_error = error_mat.squaredNorm();

	return s_error;
}

vector<MatrixXd> TE::backward(WordVec src_word_vec, WordVec& tgt_word_vec)
{
	MatrixXd derivation_mat = tgt_word_vec.word_emb - alignTable*src_word_vec.word_emb;

	vector<MatrixXd> derivations;
	derivations.push_back(derivation_mat);

	return derivations;
}