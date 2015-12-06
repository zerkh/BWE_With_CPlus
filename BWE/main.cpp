#include "Eigen/Dense"
#include <iostream>
using namespace std;
using namespace Eigen;

int main()
{
	MatrixXd m = MatrixXd::Ones(3, 3);

	m.row(0) /= m.row(0).sum();

	cout << m << endl;
}