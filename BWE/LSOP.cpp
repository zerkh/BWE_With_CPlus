#include "LSOP.h"

LSOP::LSOP(SparseMatrix<double>& alignTable)
{
	this->alignTable = alignTable;
}

double LSOP::forward(int word_pos, WordVec src_word_vec, WordVec tgt_word_vec)
{

}

vector<MatrixXd> LSOP::backward(int word_pos, WordVec src_word_vec, WordVec tgt_word_vec)
{

}