## Synthetic Populations Using Deep Generative Methods on EU-SILC Data for Finland 2013 and Norway 2017-2020

### Deep Generative Methods
Variational autoencoder and generative adversarial networks are
used to create synthetic populations from EU-SILC personal datasets.

### Data Source
EU-SILC income and living conditions data for Finland is available for downloading online from "https://ec.europa.eu/eurostat/web/microdata/public-microdata/statistics-on-income-and-living-conditions". EU-SILC for Norway is restricted and access granted through SIKT from "https://sikt.no/omrade/forskningsdata".

### General Data Preprocessing
The class Clean preprocess the EU-SILC data from the personal dataset.
DataClean. The return is one-hot-encoded categorical variables. New binary variables are created from income (hasIncome) and benefit (hasBenefits) greater than zero. Household size is reduced to five categories (householdSize), where all households with five or more members are assigned to value five. Birth year is translated into five age intervals of 13 years from 17 to 81 (17-29, 30-42, 43-55, 56-68,  69-81). The youngest in the data set are 17 years old. The fifth category additional comprises all individuals older than 81.

Missing values are imputed for variables having at least fifty per cent populated variables in the Finnish dataset and at least forty per cent in the Norwegian dataset.

### Evaluation
The synthetic populations are tested against the general and
statistic properties of the original dataset and how well the synthetic data reproduce similar results as original data on heterogeneous treatment effect analysis.

#### Metrics
* Standardised root mean squared error (SRMSE)
* Pearson's correlation coefficient
* R-squared

#### Visual evaluation
*  Marginals plots - univariate and bivariate
* Bland-Altman plots
* Contingency and confusion matrices on single or pair of variables
* Shap plots - ranked importance of variables
* Causal-tree-charts - who profits or not from intervention
* Single variables plot with metrics and binary relation to "isFemale"

#### Policy and epidemiological evaluation
* Causal Forest DML - heterogeneous treatment effects

## Tools Used
PyTorch 2.0 is the tool to set up and run deep generative models for population synthesis and clustering techniques. While Tensorflow and Keras can also be used, these tools turned out less flexible and transparent than PyTorch, especially when saving and reusing customised models with submodels like the variational autoencoder and generative adversarial network. Pandas version 1.5.3 are used for original preprocessing data in a CSV format (EU-SILC Finland) or SPSS format (EU-SILC Norway). The EU-SILC data was imputed using sklearn.impute.IterativeImputer from Scikit-Learn version 1.1.3. EconML and its model CausalForestDML version 0.13.1 are used out of the box on the observational data to generate heterogeneous treatment effects. All code is mainly run in JupyterNotbook version 6.5.4 on either a MacBookPro (without GPU) or Linux (Ubuntu 22.04 LTS) (with GPU). Visuals are created using the Seaborn package version 0.12.2, mainly based on Matplotlib-Pyplot version 3.7.1. Customised code is written to calculate, i.e. SRMSE, while Pearson's correlation coefficient and R-square are calculated from Numpy version 1.23.5 and Statsmodel version 0.13.5.

