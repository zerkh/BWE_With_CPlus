#include "Utils.h"
#include "WordVec.h"
#include "Eigen/SparseQR"
using namespace std;

void readAlignTable(SparseMatrix<double>& alignTable, string filename, WordVec src_word_vec, WordVec tgt_word_vec)
{
	ifstream in(filename.c_str(), ios::in);

	string line;
	/*
	int entities = 0;
	while (getline(in, line))
	{
	entities += 1;
	}
	in.close();
	in.open(filename.c_str(), ios::in);
	*/
	vector<Triplet<double> > tripletList;
	//	tripletList.reserve(entities);
	int count = 0;

	while (getline(in, line))
	{
		vector<string> components = splitBySpace(line);

		cout << components[0] << " " << components[1] << endl;
		cout << tgt_word_vec.m_word_id[components[0]] << " " << src_word_vec.m_word_id[components[1]] << endl;
		int tgt_id = tgt_word_vec.m_word_id[components[0]];
		int src_id = src_word_vec.m_word_id[components[1]];
		tripletList.push_back(Triplet<double>(tgt_id, src_id, (double)(atoi(components[2].c_str()) + 1)));
		cout << tripletList[count].row() << " " << tripletList[count].col() << " " << tripletList[count++].value() << endl;
	}

	alignTable.setFromTriplets(tripletList.begin(), tripletList.end());

	for (int row = 0; row < alignTable.rows(); row++)
	{
		alignTable.row(row) /= alignTable.row(row).sum();
	}

	in.close();
}


void initTgtWordVec(SparseMatrix<double>& alignTable, WordVec& src_word_vec, WordVec& tgt_word_vec)
{
	tgt_word_vec.word_emb.fill(0.0);
	tgt_word_vec.word_emb.row(0) = src_word_vec.word_emb.row(0);
	tgt_word_vec.word_emb.row(1) = src_word_vec.word_emb.row(1);

	for (int t = 0; t < tgt_word_vec.vocb_size; t++)
	{
		double sum = 0;
		if (t % 10000 == 0)
		{
			cout << "init tgt word_emb " << t << "row" << endl;
		}

		double start_clock = clock(), end_clock = 0;
		for (int s = 0; s < src_word_vec.vocb_size; s++)
		{
			if (s % 10000 == 0)
			{
				end_clock = clock();
				cout << sum << endl;
				cout << "the " << s << " src_emb. The cost of time is " << (end_clock - start_clock) / CLOCKS_PER_SEC << endl;
				start_clock = clock();
			}
			tgt_word_vec.word_emb.row(t) += alignTable.coeffRef(t,s)*src_word_vec.word_emb.row(s);
		}
	}
}

int main()
{
	Config conf("Config.conf");
	SparseMatrix<double> alignTable;
	double start_clock, end_clock;

	//get config
	string src_vocab_file = conf.get_para("src_vocab_file");
	string tgt_vocab_file = conf.get_para("tgt_vocab_file");
	int word_dim = atoi(conf.get_para("word_dim").c_str());
	string output_dir = conf.get_para("output_dir");
	string src_align_table_file = conf.get_para("src_align_table");
	string tgt_align_table_file = conf.get_para("tgt_align_table");

	//init word vectors
	WordVec src_word_vec(word_dim, src_vocab_file);
	WordVec tgt_word_vec(word_dim, tgt_vocab_file);
	alignTable = SparseMatrix<double>(tgt_word_vec.vocb_size, src_word_vec.vocb_size);

	//load word vector of source language pre-trained by GCWE
	cout << "Load src word vectors from \"" << conf.get_para("src_word_vec") << "\"......" << endl;
	src_word_vec.loadWordVec(conf.get_para("src_word_vec"));

	//init target word vector with equivalence and source word vector
	start_clock = clock();
	cout << "Reading alignment table......" << endl;
	readAlignTable(alignTable, tgt_align_table_file, src_word_vec, tgt_word_vec);
	end_clock = clock();
	cout << "Complete to read alignment table! The cost of time is " << (end_clock - start_clock) / CLOCKS_PER_SEC << endl;

	cout << "Initializing target word vectors......" << endl;
	start_clock = clock();
	initTgtWordVec(alignTable, src_word_vec, tgt_word_vec);
	end_clock = clock();
	cout << "Complete to initialize target word vectors! The cost of time is " << (end_clock - start_clock) / CLOCKS_PER_SEC << endl;

	//Save
	tgt_word_vec.saveWordVec(output_dir, "tgt");

	return 0;
}
