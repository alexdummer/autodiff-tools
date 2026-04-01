#include "autodiff/forward/dual.hpp"
#include <functional>

void makeDualArray(const double T[3][3], autodiff::dual T_dual[3][3]) {
  using namespace autodiff;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      T_dual[i][j] =
          dual(T[i][j]); // Initialize with the value and zero derivative
    }
  }
}

void makeDual2ndArray(const double T[3][3], autodiff::dual2nd T_dual[3][3]) {

  using namespace autodiff;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      T_dual[i][j] =
          dual2nd(T[i][j]); // Initialize with the value and zero derivatives
    }
  }
}

void df_dT(std::function<autodiff::dual(const autodiff::dual[3][3])> f,
           const double T[3][3], double df_dT_out[3][3]) {

  using namespace autodiff;
  // initialize an array with dual numbers
  dual T_dual[3][3];
  makeDualArray(T, T_dual);

  // Extract the derivatives from the dual numbers
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      seed<1>(T_dual[i][j],
              1.0); // Set the derivative with respect to T[i][j] to 1
      dual f_dual = f(T_dual); // Evaluate the function with dual numbers
      df_dT_out[i][j] =
          derivative<1>(f_dual);  // Get the derivative with respect to T[i][j]
      seed<1>(T_dual[i][j], 0.0); // Reset the derivative for the next iteration
    }
  }
}

void d2f_dT2(std::function<autodiff::dual2nd(const autodiff::dual2nd[3][3])> f,
             const double T[3][3], double d2f_dT2_out[3][3][3][3]) {
  using namespace autodiff;
  // initialize an array with dual numbers
  dual2nd T_dual[3][3];
  makeDual2ndArray(T, T_dual);

  // Extract the second derivatives from the dual2nd numbers
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      // Set the first derivative with respect to T[i][j] to 1
      seed<1>(T_dual[i][j], 1);
      for (int k = 0; k < 3; ++k) {
        for (int l = 0; l < 3; ++l) {
          // Set the second derivative with respect to T[k][l] to 1
          seed<2>(T_dual[k][l], 1);
          // Evaluate the function with dual2nd numbers
          dual2nd f_dual = f(T_dual);
          // Get the second derivative with respect to T[i][j] and T[k][l]
          d2f_dT2_out[i][j][k][l] = derivative<2>(f_dual);
          // Reset the second derivative for the next iteration
          seed<2>(T_dual[k][l], 0);
        }
      }
      // Reset the first derivative for the next iteration
      seed<1>(T_dual[i][j], 0);
    }
  }
}
