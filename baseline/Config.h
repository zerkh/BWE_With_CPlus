#ifndef __CONFIG__
#define __CONFIG__

#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <sstream>
#include "Utils.h"
using namespace std;

class Config
{
private:
	map<string, string> m_para;
	vector<string> v_para_str;
public:
	Config(const string& config_file) 
	{
		cout << "Reading parameters from \"" << config_file << "\"" << endl << endl;

		ifstream is(config_file.c_str());
		if (!is) {
			cerr << "Parameter Error, fail to read config file from \"" << config_file << "\"." << endl;
			exit(1);
		}

		string line, title_str, equal_str, value_str;
		size_t pos;
		while (getline(is, line)) 
		{
			if (line == "" || (pos = line.find("=")) == string::npos) 
			{
				continue;
			}

			title_str = strip_str(line.substr(0, pos));
			pos++;
			value_str = strip_str(line.substr(pos));

			m_para[title_str] = value_str;
			v_para_str.push_back(title_str + "\t\t" + value_str);
		}

		is.close();
	};

	string get_para(const string& title_str) 
	{
		map<string, string>::iterator m_it = m_para.find(title_str);
		if (m_it == m_para.end())
		{
			cerr << "Get_para error, fail to read the value of item \"" << title_str << "\"." << endl;
			exit(1);
		}

		return m_it->second;
	};

	void set_para(const string& input_item, const string& input_value_str) 
	{
		m_para[input_item] = input_value_str;
	};

	void show_para() 
	{
		cout << "The parameters are shown as follows: " << endl;
		for (size_t i = 0; i < v_para_str.size(); i++) 
		{
			cout << v_para_str[i] << endl;
		}
		cout << endl << endl;
	}
};
#endif