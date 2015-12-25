#include "TE.h"

TE::TE(WordVec src_word_vec, WordVec tgt_word_vec)
{
	src_vocb_size = src_word_vec.vocb_size;
	tgt_vocb_size = tgt_word_vec.vocb_size;

	alignTable = SparseMatrix<double>(tgt_vocb_size, src_vocb_size);
}

/*
input: 
filename: example: {alignTable.xml} format: {src_word tgt_word align_times}
*/
void TE::readAlignTable(string filename, WordVec src_word_vec, WordVec tgt_word_vec)
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
		tripletList.push_back(Triplet<double>(tgt_id, src_id, (double)(atoi(components[2].c_str()) + 1)) );
		cout << tripletList[count].row() << " " << tripletList[count].col() << " " << tripletList[count++].value() << endl;
	}

	alignTable.setFromTriplets(tripletList.begin(), tripletList.end());

	for (int row = 0; row < alignTable.rows(); row++)
	{
		alignTable.row(row) /= alignTable.row(row).sum();
	}

	in.close();
}

void TE::initTgtWordVec(WordVec src_word_vec, WordVec& tgt_word_vec)
{
	tgt_word_vec.word_emb.fill(0.0);

	for (int t = 0; t < tgt_vocb_size; t++)
	{
		double sum = 0;
		if(t % 10000 == 0)
		{
			cout << "init tgt word_emb " << t << "row" << endl;
		}

		double start_clock = clock(), end_clock = 0;
		for (int s = 0; s < src_vocb_size; s++)
		{
			if(s % 10000 == 0)
			{
				end_clock = clock();
				cout << sum << endl;
				cout << "the " << s << " src_emb. The cost of time is " << (end_clock-start_clock) / CLOCKS_PER_SEC << endl;
				start_clock = clock();
			}
			//tgt_word_vec.word_emb.row(t) += alignTable.coeffRef(t,s)*src_word_vec.word_emb.row(s);
			sum += alignTable.coeffRef(t,s);
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
