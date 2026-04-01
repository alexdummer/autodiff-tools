#include "autodiff/forward/dual.hpp"
#include "interface.h"
#include "neohooke.h"
#include <iostream>
int main() {

  using namespace autodiff;
  // material parameters
  double K = 3500;
  double G = 1500;

  // right Cauchy green deformation tensor
  double C[3][3];
  C[0][0] = 1.0;
  C[0][1] = 0.0;
  C[0][2] = 0.0;
  C[1][0] = 0.0;
  C[1][1] = 1.0;
  C[1][2] = 0.0;
  C[2][0] = 0.0;
  C[2][1] = 0.0;
  C[2][2] = 1.0;

  // lambda to pass K and K to the energy density function
  auto energy_density = [&](const dual T[3][3]) {
    dual nrg = psi(T, K, G);
    return nrg;
  };

  auto energy_density_2nd = [&](const dual2nd T[3][3]) {
    dual2nd nrg = psi(T, K, G);
    return nrg;
  };

  double dPsi_dC_out[3][3];
  df_dT(energy_density, C, dPsi_dC_out);

  std::cout << "Derivatives df/dT:" << std::endl;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      std::cout << "dPsi/dC[" << i << "][" << j << "] = " << dPsi_dC_out[i][j]
                << std::endl;
    }
  }

  double d2Psi_dC2_out[3][3][3][3];
  d2f_dT2(energy_density_2nd, C, d2Psi_dC2_out);

  std::cout << "Second derivatives d2f/dT2:" << std::endl;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        for (int l = 0; l < 3; ++l) {
          std::cout << "d2Psi/dC[" << i << "][" << j << "]/dC[" << k << "]["
                    << l << "] = " << d2Psi_dC2_out[i][j][k][l] << std::endl;
        }
      }
    }
  }
  return 0;
}
