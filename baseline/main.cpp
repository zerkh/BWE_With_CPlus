#include "Eigen/Dense"
#include <iostream>
#include "Config.h"
#include "Utils.h"
#include "GCWE.h"
#include "TE.h"
#include "WordVec.h"
using namespace std;
using namespace Eigen;

/*
Baseline of our work
Reference Zou EMNLP 2013
Author kh
*/

void sgd_one_step()
{

}

int main()
{
	Config conf("Config.conf");

	string src_vocab_file = conf.get_para("src_vocab_file");
	string tgt_vocab_file = conf.get_para("tgt_vocab_file");
	int word_dim = atoi(conf.get_para("word_dim").c_str());

	WordVec src_word_vec(word_dim, src_vocab_file);
	WordVec tgt_word_vec(word_dim, tgt_vocab_file);

	src_word_vec.loadWordVec(conf.get_para("src_word_vec"));
}