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
	int word_dim;
	int vocb_size;

	WordVec(int word_dim, int vocb_size);
};