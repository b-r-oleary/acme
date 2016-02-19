# acme
a repository of code written for the ACME collaboration

Here is a list of the contents:

### Statistics

This is my statistics package that is mainly located in `./Statistics/code/statfunctions.py` which includes:
- some functions applicable to use with normally distributed random variables with different variances. This data generally consists of mean estimates and error estimates.
  - `weighted_mean`
  - `unweighted_mean`
  - `chi_squared`
  - `correlation_coefficient`
  - `autocorrelation`
  - `scientific_string`
  - `t_hist`
  - `linear_regression`
  - `linear_regression_with_linear_constraints`
  - `normal_p_to_n_sigma`
  - `normal_n_sigma_to_p`
- some classes that enable various types of linear and nonlinear regression and plotting
  - `LinearRegression`
  - `NonLinearFit1D`
  - `GaussianFit1D`
  - `LinearRegression2D`
  - `PolynomialFit2D`
  - `LinearRegression1D`
  - `PolynomialFit1D`
  - `SineFit1D`
  - `FourierTransform`
- and some plotting methods
  - `regression_plot`
  - `plot3D`

