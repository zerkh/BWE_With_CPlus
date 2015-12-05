#pragma once
#include "Eigen/Dense"
#include <iostream>
#include <map>
using namespace std;
using namespace Eigen;

class WordVec
{
public:
	MatrixXd word_emb;
	map<string, int> m_word_id;
	map<int, double> m_id_idf;
	int word_dim;
	int vocb_size;

	WordVec(int word_dim, int vocb_size)
	{
		this->word_dim = word_dim;
		this->vocb_size = vocb_size;

		word_emb = MatrixXd::Random(vocb_size, word_dim);
	}
};
