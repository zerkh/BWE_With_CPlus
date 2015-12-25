#ifndef __TE__
#define __TE__

#include "Eigen/Core"
#include "WordVec.h"
#include "GCWE.h"
#include "Utils.h"
#include <ctime>
#include <iostream>
#include "Eigen/SparseQR"

using namespace std;
using namespace Eigen;

class TE
{
public:
	SparseMatrix<double> alignTable;
	int src_vocb_size;
	int tgt_vocb_size;

	TE() {};

	TE(WordVec src_word_vec, WordVec tgt_word_vec);

	void readAlignTable(string filename, WordVec src_word_vec, WordVec tgt_word_vec);

	void initTgtWordVec(WordVec src_word_vec, WordVec& tgt_word_vec);

	double forward(WordVec src_word_vec, WordVec tgt_word_vec);

	vector<MatrixXd> backward(WordVec src_word_vec, WordVec& tgt_word_vec);
};

#endif
