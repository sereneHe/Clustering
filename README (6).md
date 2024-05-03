# Joint Problems in Learning Multiple Dynamical Systems

This is the source code for the paper Joint Problems in Learning Multiple Dynamical Systems.

It includes the notebooks and data files for each figure in the paper.

## Get Started
1. Dataset: "1_data_generation.ipynb" can be used to generate your own dataset. You can also utilize generated data in the folder ./Synthetic data and real-world ECG data in ./ECG5000. You can download the ECG data from https://www.timeseriesclassification.com/description.php?Dataset=ECG5000.
2. Experiments: We provide the experiment scripts of both proposed methods and baselines. You can access them through jupyter notebooks with the prefix "2_".
3. Results: All results and figures mentioned in the paper are under the folder ./result. You can utilize "3_plot.ipynb" to visualize the results.

## Run Notebooks
1. 2_EM_MIP-IF.ipynb: This notebook contains scripts for all experiments on both synthetic and real-world data using MIP-IF and EM Heuristic approaches mentioned in our paper. You can upload datasets and the notebook to Colab https://colab.research.google.com/ and run it.
2. 2_EM-NCPOP.ipynb: This notebook contains essential scripts for additional experiments mentioned in our supplementary text. We run the notebook on Colab. If you use Mosek as a solver, a license is required. After applying for a license from Mosek, you can put "mosek.lic" to colab Files.
3. 2_method_DTW/FFT.ipynb: These two notebooks are used for baseline tests. You can easily run them with your own preference.


If one would like to run experiments on local:
## Dependencies
1. Python>=3.9.7
2. Pyomo v6.6.2 https://www.pyomo.org/
3. Bonmin https://www.coin-or.org/Bonmin/
4. ncpol2sdpa 1.12.2 https://ncpol2sdpa.readthedocs.io/en/stable/index.html
5. Mosek 10.1 https://www.mosek.com/
