#include "Eigen/Core"
#include <iostream>
using namespace std;
using namespace Eigen;

class GCWE
{
public:
	MatrixXd W1;
	VectorXd b1;
	MatrixXd W2;
	VectorXd b2;
	MatrixXd Wg1;
	VectorXd bg1;
	MatrixXd Wg2;
	VectorXd bg2;
	int word_dim, hidden_dim, window_size;

	GCWE(int word_dim, int hidden_dim, int window_size)
	{
		this->word_dim = word_dim;
		this->hidden_dim = hidden_dim;
		this->window_size = window_size;

		W1 = MatrixXd::Random(window_size*word_dim, hidden_dim);
		b1 = VectorXd::Zero(hidden_dim);
		W2 = MatrixXd::Random(hidden_dim, 1);
		b2 = VectorXd::Zero(1, 1);

		Wg1 = MatrixXd::Random(2 * word_dim, hidden_dim);
		bg1 = VectorXd::Zero(hidden_dim);
		Wg2 = MatrixXd::Random(hidden_dim, 1);
		bg2 = VectorXd::Zero(1, 1);
	}


};