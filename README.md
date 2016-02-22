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

## MIRGInterferometer

This is my some analysis code for scans taken with the white light interferometer. The code is mainly located in `./MIRGInterferometer/code/interferometer_data_manipulation.py` which includes:

- `InterferometerData`: a class used to import and analyze interferometer scans.
- `InterferometerMap`: a class that fits a series of interferometer scans to a 2D polynomial to obtain a 2D map.

The data is kept in `./MIRGInterferometer/data`, and output plots are saved to `./MIRGInterferometer/outputs`.

## MassSpectra

This is my analysis code for scans taken with the RGA (residual gas analyzer) for determining the gas composition within the vacuum chamber (mostly used while baking the chamber). The code is located in `./MassSpectra/code/mass_spectra.py`. A library of mass spectra taken from the NIST database is stored in `./MassSpectra/library`. RGA scans are stored in `./MassSpectra/data`, and outputs are also pushed to that directory. The code consists of a couple classes:

- `MassSpectrumLibrary`: used to import from `.jdx` and store a library of mass spectra
- `MassSpectrum`: used to import RGA scans from `.xml` and analyze a scan with respect to a mass spectrum library.

## Simulations

This will be a repository of simulations. For right now, I just have very rudimentary Schrodinder Equation Integration code in `./Simulations/code/simulations.py` which introduces a class `IntegrateSchrodingerEquation`.

## MoleculeCalculations

This is a repository of tools used to perform calculations with diatomic molecules, in particular ThO. There is a code repository . `./MoleculeCalculations/code` within which there are a few files:

- `matrix_elements.py`: which has some function definitions for calculating matrix elements (Mostly from chapter 5 of Brown and Carrington)
- `molecule.py`: which introduces a `State` class that describes a molecular electronic state, and a `Molecule` class that contains a list of States and methods to evaluate transtion frequencies and plot energy level diagrams.

## DatabaseAccess

This is a repository of tools for accessing the ACME database. This introduces the database object `DatabaseAccess` that is built on top of a `pyodbc` connection. Data can be easily extracted and is put into `TimeSeriesArray` objects which enable easy plotting and saving.

## SymbolicManipulation

This is a repository for add-ons to `sympy` a symbolic manipulation package for python. The file `./SymbolicManipulation/code/sympy_plotting.py` contains two methods `plots` and `manipulate` that extends the sympy plotting functionality to include a mathematica-like manipulate capabilities within an ipython notebook.
