#pragma once
#include "Eigen/Dense"
#include <iostream>
#include <map>
#include <ctime>
#include <string>
#include <sstream>
#include <vector>
#include <fstream>
#include "Utils.h"
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
		int cur_id = 1;
		m_word_id["S"] = 0;
		m_id_word[0] = "S";

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

	void loadWordVec(string filename)
	{
		ifstream in(filename.c_str(), ios::in);

		m_id_word.clear();
		m_word_id.clear();
		int cur_id = 0;

		string line;
		getline(in, line);

		//load vocb_size, word_dim
		vector<string> first_line = splitBySpace(line);
		vocb_size = atoi(first_line[0].c_str());
		word_dim = atoi(first_line[1].c_str());

		word_emb.resize(vocb_size, word_dim);

		//load word vectors
		while (getline(in, line))
		{
			vector<string> word_vec = splitBySpace(line);

			m_id_word[cur_id] = word_vec[0];
			m_word_id[word_vec[0]] = cur_id;

			for (int i = 0; i < word_dim; i++)
			{
				word_emb(cur_id, i) = stod(word_vec[i + 1]);
			}
		}
	}

	void saveWordVec(string output_dir)
	{
		time_t t = time(0);

		struct tm* now = localtime(&t);
		string mon = string(itoa(now->tm_mon + 1));
		string day = string(itoa(now->tm_mday));
		string hour = string(itoa(now->tm_hour));

		string filename = output_dir + mon + "-" + day + "-" + hour + "-src.vec";
	
		ofstream out(filename.c_str(), ios::out);

		out << vocb_size << " " << word_dim << endl;

		for (int row = 0; row < vocb_size; row++)
		{
			out << m_id_word[row] << " ";

			for (int col = 0; col < vocb_size; col++)
			{
				if (col != vocb_size - 1)
				{
					out << word_emb(row, col) << " ";
				}
				else
				{
					out << word_emb(row, col);
				}
			}

			out << endl;
		}

		out.close();
	}
};
