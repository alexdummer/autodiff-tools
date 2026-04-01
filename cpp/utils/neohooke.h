#pragma once
#include <cmath>

template <typename T>
T trace(const T C[3][3]) { // compute the trace of a 3x3 matrix C
  return C[0][0] + C[1][1] + C[2][2];
}

template <typename T> T determinant(const T C[3][3]) {
  // compute the determinant of a 3x3 matrix C
  return C[0][0] * (C[1][1] * C[2][2] - C[1][2] * C[2][1]) -
         C[1][0] * (C[0][1] * C[2][2] - C[0][2] * C[2][1]) +
         C[2][0] * (C[0][1] * C[1][2] - C[0][2] * C[1][1]);
}

template <typename T> T psi(const T C[3][3], const double K, const double G) {

  const T J = sqrt(determinant(C));
  const T I1 = trace(C);

  T res = K / 8. * pow(J - 1. / J, 2.) + G / 2. * (I1 * pow(J, -2. / 3) - 3.);

  return res;
}
