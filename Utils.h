#pragma once

#include "Eigen/Core"
#include <iostream>

RowVectorXd tanh(RowVectorXd v)
{
	RowVectorXd v_tanh(v.cols());

	for (int i = 0; i < v.cols(); i++)
	{
		v_tanh(i) = (exp(v(i) * 2) - 1) / (exp(v(i) * 2) + 1);
	}

	return v_tanh;
}

RowVectorXd derTanh(RowVectorXd v)
{
	RowVectorXd v_derTanh = mulByElem(v, v);

	for (int i = 0; i < v.cols(); i++)
	{
		v_derTanh(i) = 1 - v_derTanh(i);
	}

	return v_derTanh;
}

MatrixXd mulByElem(MatrixXd m1, MatrixXd m2)
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