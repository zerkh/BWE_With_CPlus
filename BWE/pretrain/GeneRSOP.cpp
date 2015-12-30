#include "Eigen/Core"
#include <iostream>
#include <vector>
#include "Utils.h"
#include "Config.h"
using namespace std;
using namespace Eigen;

class PhrasePair
{
public:
	string src_phrase;
	string tgt_phrase;

	vector<string> v_src_phrase;
	vector<string> v_tgt_phrase;

	map<int, int> s2t_alignment;
	map<int, int> t2s_alignment;

	double tp_s2t;
	double tp_t2s;

	PhrasePair()
	{};

	PhrasePair(string line)
	{
		vector<string> components = splitString(line, " || ");

		src_phrase = components[0];
		tgt_phrase = components[1];
		v_src_phrase = splitBySpace(src_phrase);
		v_tgt_phrase = splitBySpace(tgt_phrase);

		vector<string> datas = splitBySpace(components[2]);

		tp_t2s = atof(datas[0].c_str());
		tp_s2t = atof(datas[2].c_str());

		vector<string> v_align_seg = splitBySpace(components[3]);
		for (int i = 0; i < v_align_seg.size(); i++)
		{
			s2t_alignment[atoi(splitString(v_align_seg[i], "-")[0].c_str())] = atoi(splitString(v_align_seg[i], "-")[1].c_str());
			t2s_alignment[atoi(splitString(v_align_seg[i], "-")[1].c_str())] = atoi(splitString(v_align_seg[i], "-")[0].c_str());
		}
	}

	void transferToVector()
	{
		v_src_phrase = splitBySpace(src_phrase);
		v_tgt_phrase = splitBySpace(tgt_phrase);
	}
};

class GeneRSOP
{
public:
	vector<PhrasePair> src_phrase_alignment;
	vector<PhrasePair> tgt_phrase_alignment;
	map<pair<string, string>, double> src_word_alignment;
	map<pair<string, string>, double> tgt_word_alignment;
	map<pair<string, string>, double> src_count;
	map<pair<string, string>, double> tgt_count;
	vector<PhrasePair> v_phrasepair;

	vector<string> src_phrase_table;
	vector<string> tgt_phrase_table;
	map<string, vector<int> > m_src_phrasepair;
	map<string, vector<int> > m_tgt_phrasepair;
	map<string, vector<string> > m_sp2tp;
	map<string, vector<string> > m_tp2sp;

	GeneRSOP(string phrase_table_file)
	{
		ifstream in(phrase_table_file);

		string line;
		while (getline(in, line))
		{
			PhrasePair pp = PhrasePair(line);
			v_phrasepair.push_back(pp);

			m_src_phrasepair[pp.src_phrase].push_back(v_phrasepair.size() - 1);
			m_tgt_phrasepair[pp.tgt_phrase].push_back(v_phrasepair.size() - 1);

			m_sp2tp[pp.src_phrase].push_back(pp.tgt_phrase);
			m_tp2sp[pp.tgt_phrase].push_back(pp.src_phrase);

			vector<string>::iterator it = find(src_phrase_table.begin(), src_phrase_table.end(), pp.src_phrase);
			if (it == src_phrase_table.end())
			{
				src_phrase_table.push_back(pp.src_phrase);
			}

			it = find(tgt_phrase_table.begin(), tgt_phrase_table.end(), pp.tgt_phrase);
			if (it == tgt_phrase_table.end())
			{
				tgt_phrase_table.push_back(pp.tgt_phrase);
			}
		}

		in.close();
	};

	void geneMonoAlign()
	{
		//get source and target phrase alignment
		for (int first = 0; first < src_phrase_table.size(); first++)
		{
			for (int second = first + 1; second < src_phrase_table.size(); second++)
			{
				int count = 0;
				double tp_s2t = 0, tp_t2s = 0;
				vector<int> v_tmp_sp2tp1 = m_src_phrasepair[src_phrase_table[first]];
				vector<int> v_tmp_sp2tp2 = m_src_phrasepair[src_phrase_table[second]];

				for (int i = 0; i < v_tmp_sp2tp1.size(); i++)
				{
					for (int j = 0; j < v_tmp_sp2tp2.size(); j++)
					{
						if (v_phrasepair[v_tmp_sp2tp1[i]].tgt_phrase == v_phrasepair[v_tmp_sp2tp2[j]].tgt_phrase)
						{
							count++;
							PhrasePair pp;
							tp_s2t = v_phrasepair[v_tmp_sp2tp1[i]].tp_s2t*v_phrasepair[v_tmp_sp2tp2[j]].tp_t2s;
							tp_t2s = v_phrasepair[v_tmp_sp2tp2[i]].tp_s2t*v_phrasepair[v_tmp_sp2tp1[j]].tp_t2s;
							pp.src_phrase = src_phrase_table[first];
							pp.tgt_phrase = src_phrase_table[second];
							pp.tp_s2t = tp_s2t;
							pp.tp_t2s = tp_t2s;
							pp.transferToVector();

							map<int, int> s2t_align = v_phrasepair[v_tmp_sp2tp1[i]].s2t_alignment;
							map<int, int> t2s_align = v_phrasepair[v_tmp_sp2tp2[i]].t2s_alignment;
							for (map<int, int>::iterator it = s2t_align.begin(); it != s2t_align.end(); it++)
							{
								if (t2s_align.find(it->second) != t2s_align.end())
								{
									pp.s2t_alignment[it->first] = t2s_align[it->second];
									pp.t2s_alignment[t2s_align[it->second]] = it->first;
								}
							}

							src_phrase_alignment.push_back(pp);
						}
					}
				}
			}
		}

		for (int first = 0; first < tgt_phrase_table.size(); first++)
		{
			for (int second = first + 1; second < tgt_phrase_table.size(); second++)
			{
				int count = 0;
				double tp_s2t = 0, tp_t2s = 0;
				vector<int> v_tmp_sp2tp1 = m_tgt_phrasepair[tgt_phrase_table[first]];
				vector<int> v_tmp_sp2tp2 = m_tgt_phrasepair[tgt_phrase_table[second]];

				for (int i = 0; i < v_tmp_sp2tp1.size(); i++)
				{
					for (int j = 0; j < v_tmp_sp2tp2.size(); j++)
					{
						if (v_phrasepair[v_tmp_sp2tp1[i]].src_phrase == v_phrasepair[v_tmp_sp2tp2[j]].src_phrase)
						{
							count++;
							PhrasePair pp;
							tp_s2t = v_phrasepair[v_tmp_sp2tp1[i]].tp_t2s*v_phrasepair[v_tmp_sp2tp2[j]].tp_s2t;
							tp_t2s = v_phrasepair[v_tmp_sp2tp2[i]].tp_t2s*v_phrasepair[v_tmp_sp2tp1[j]].tp_s2t;
							pp.src_phrase = tgt_phrase_table[first];
							pp.tgt_phrase = tgt_phrase_table[second];
							pp.tp_s2t = tp_s2t;
							pp.tp_t2s = tp_t2s;
							pp.transferToVector();

							map<int, int> s2t_align = v_phrasepair[v_tmp_sp2tp1[i]].s2t_alignment;
							map<int, int> t2s_align = v_phrasepair[v_tmp_sp2tp2[i]].t2s_alignment;
							for (map<int, int>::iterator it = s2t_align.begin(); it != s2t_align.end(); it++)
							{
								if (t2s_align.find(it->second) != t2s_align.end())
								{
									pp.s2t_alignment[it->first] = t2s_align[it->second];
									pp.t2s_alignment[t2s_align[it->second]] = it->first;
								}
							}

							tgt_phrase_alignment.push_back(pp);
						}
					}
				}
			}
		}
	}

	void calcWordRepeat(string src_vocb_file, string tgt_vocb_file)
	{
		vector<string> v_src_vocb;
		vector<string> v_tgt_vocb;

		ifstream src_in(src_vocb_file.c_str(), ios::in);
		ifstream tgt_in(tgt_vocb_file.c_str(), ios::in);

		string line;

		while (getline(src_in, line))
		{
			vector<string> components = splitString(line, string("\t"));
			v_src_vocb.push_back(components[1]);
		}

		while (getline(tgt_in, line))
		{
			vector<string> components = splitString(line, string("\t"));
			v_tgt_vocb.push_back(components[1]);
		}

		for (int first = 0; first < v_src_vocb.size(); first++)
		{
			for (int second = first + 1; second < v_src_vocb.size(); second++)
			{
				pair<string, string> pss = pair<string, string>(v_src_vocb[first], v_src_vocb[second]);
				src_count[pss] = 0;
			}
		}

		src_in.close();
		tgt_in.close();
	}
};