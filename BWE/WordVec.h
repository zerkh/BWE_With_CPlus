#pragma once
#include "Eigen/Dense"
#include <iostream>
#include <map>
#include <vector>
#include <fstream>
using namespace std;
using namespace Eigen;

class WordVec
{
public:
	MatrixXd word_emb;
	map<string, int> m_word_id;
	map<int, string> m_id_word;
	vector<double> v_id_idf;
	int word_dim;
	int vocb_size;

	WordVec(int word_dim, string filename)
	{
		this->word_dim = word_dim;

		cout << "Init vocabulary......." << endl;
		init(filename);

		this->vocb_size = m_word_id.size();

		word_emb = MatrixXd::Random(vocb_size, word_dim);

		for(int i = 0; i < vocb_size; i++)
		{
			v_id_idf.push_back(1);
		}
	}

	//need en/ch.vcb, 1	word	occurrence
	void init(string filename)
	{
		ifstream in(filename.c_str(), ios::in);
		int cur_id = 0;

		string line;
		while (getline(in, line))
		{
			vector<string> components = splitString(line, string("\t"));

			m_word_id[components[1]] = cur_id;
			m_id_word[cur_id] = components[1];
		}
	}

	//need raw corpus	
	void init_idf(string filename)
	{
		ifstream in(filename.c_str(), ios::in);
		vector<string> sentences;

		string line;
		while (getline(in, line))
		{
			sentences.push_back(line);
		}

		for (int w = 0; w < vocb_size; w++)
		{
			for (int s = 0; s < sentences.size(); s++)
			{
				if (sentences[s].find(m_id_word[w]) != string::npos)
				{
					v_id_idf[w] += 1;
				}
			}

			v_id_idf[w] = sentences.size() / v_id_idf[w];
		}
	}
};
