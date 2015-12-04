#pragma once

#include "Eigen/Core"
#include <iostream>

VectorXd tanh(VectorXd v)
{
	VectorXd v_tanh(v.cols());

	for (int i = 0; i < v.cols(); i++)
	{
		v_tanh(i) = (exp(v(i) * 2) - 1) / (exp(v(i) * 2) + 1);
	}

	return v_tanh;
}
