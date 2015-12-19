#ifndef __UTILS__
#define __UTILS__

#include "Eigen/Core"
#include <iostream>
#include <vector>
#include <sstream>
#include <string>
#include "Eigen/SparseQR"
#include <pthread.h>
using namespace std;
using namespace Eigen;

inline string strip_str(const string& s, const string chs = " \t\n")
{
	if (s.empty()) {
		return s;
	}

	int i = 0;
	while (i < (int)s.size() && chs.find(s[i]) != string::npos) {
		++i;
	}

	int j = (int)s.size() - 1;
	while (j >= 0 && chs.find(s[j]) != string::npos) {
		--j;
	}

	++j;

	return i >= j ? "" : s.substr(i, j - i);
};

/*
inline string to_string(int i)
{
	vector<char> pos;
	while (i / 10 != 0)
	{
		pos.push_back(i % 10 + '0');
		i /= 10;
	}
	pos.push_back(i + '0');

	string str = "";

	for (int i = pos.size() - 1; i >= 0; i--)
	{
		str += pos[i];
	}

	return str;
}
*/

inline RowVectorXd tanh(RowVectorXd v)
{
	RowVectorXd v_tanh(v.cols());

	for (int i = 0; i < v.cols(); i++)
	{
		v_tanh(i) = (exp(v(i) * 2) - 1) / (exp(v(i) * 2) + 1);
	}

	return v_tanh;
}

inline MatrixXd mulByElem(MatrixXd m1, MatrixXd m2)
{
	MatrixXd m(m1.rows(), m1.cols());

	for (int row = 0; row < m.rows(); row++)
	{
		for (int col = 0; col < m.cols(); col++)
		{
			m(row, col) = m1(row, col) * m2(row, col);
		}
	}

	return m;
}

inline RowVectorXd derTanh(RowVectorXd v)
{
	RowVectorXd v_derTanh = mulByElem(v, v);

	for (int i = 0; i < v.cols(); i++)
	{
		v_derTanh(i) = 1 - v_derTanh(i);
	}

	return v_derTanh;
}

inline vector<string> splitBySpace(string line)
{
	vector<string> words;
	stringstream ss(line);

	string word;
	while (ss >> word)
	{
		words.push_back(word);
	}

	return words;
}

inline vector<string> splitString(string& s, const string c)
{
	vector<string> v;
	string::size_type pos1, pos2;
	pos2 = s.find(c);
	pos1 = 0;

	while (string::npos != pos2)
	{
		v.push_back(s.substr(pos1, pos2 - pos1));

		pos1 = pos2 + c.size();
		pos2 = s.find(c, pos1);
	}

	if (pos1 != s.length())
		v.push_back(s.substr(pos1));

	return v;
}

#endif
