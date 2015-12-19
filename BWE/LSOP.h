#ifndef __LSOP__
#define __LSOP__
#include "Utils.h"
#include "WordVec.h"
#include "Eigen/SparseQR"

class LSOP
{
public:
	SparseMatrix<double> alignTable;

	LSOP(SparseMatrix<double>& alignTable);

	double forward(int word_pos, WordVec src_word_vec, WordVec tgt_word_vec);

	vector<MatrixXd> backward(int word_pos, WordVec src_word_vec, WordVec tgt_word_vec);
};

#endif __LSOP__
