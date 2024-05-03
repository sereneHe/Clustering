# Clustering Problems in Learning Joint Multiple Dynamical Systems
Welcome to the repository for clustering Multiple Linear Dynamical Systems(LDS)! This codebase encompasses notebooks and test results associated with figures and results in the [Paper](https://arxiv.org/abs/2311.02181).

## Joint-Multiple LDSs Learning Process
1. **Generate Data:** Use the `data_generation` function in `EM_Heuristic_Data_Generation.ipynb` to create artificial observation data.
2. **Learn LDS Structure:** Uncover the Joint Multiple Dynamical Systems beneath the observation data.
3. **Cluster multi-Data:** Use MIP-IF and EM Heuristic Methods to cluster multiple dynamical systems
4. **Visualize Comparison:** Illustrate F1 results in a box plot for multiple datasets.

## Getting Started
1. **Synthetic Data:** Utilize the `EM_Heuristic_Data_Generation.ipynb` anaconda notebook to generate custom datasets. Alternatively, use generated data in `./Data/Synthetic data`.
2. **Real-world Data:** Employ the `EM_Heuristic_Real_Test` class in `EM_Heuristic_Real_Test.ipynb` to test real data from the `./Data/Real data` folder
3. **Experiments:** Experiment scripts for proposed methods and baselines are available in Jupyter notebooks named "Methods_Baseline.ipynb".
4. **Results:** Access results and figures mentioned in the paper from the `./result` folder. Repeat the results using "EM_Heuristic_test.ipynb".

## Running Notebooks
- **EM_Heuristic_test.ipynb:** Execute this notebook for all experiments on synthetic and real-world data using the EM_Heuristic approach. Upload datasets and the notebook to [Colab](https://colab.research.google.com/) for seamless execution.

## Installation
To run experiments locally, ensure you have the following dependencies installed:
- Python (>= 3.6, <= 3.9)
- tqdm (>= 4.48.2)
- NumPy (>= 1.19.1)
- Pandas (>= 0.22.0)
- SciPy (>= 1.7.3)
- scikit-learn (>= 0.21.1)
- Matplotlib (>= 2.1.2)
- NetworkX (>= 2.5)
- PyTorch (>= 1.9.0)
- ncpol2sdpa 1.12.2 [Documentation](https://ncpol2sdpa.readthedocs.io/en/stable/index.html)
- MOSEK (>= 9.3) [MOSEK](https://www.mosek.com/)
- gcastle (>= 1.0.3)

### PIP Installation
```bash
# Execute the following commands to run the notebook directly in Colab. Ensure your MOSEK license file is in one of these locations:
#
# /content/mosek.lic   or   /root/mosek/mosek.lic
#
# inside this notebook's internal filesystem.
# Install MOSEK and ncpol2sdpa if not already installed
pip install mosek 
pip install ncpol2sdpa
pip install gcastle==1.0.3
```
