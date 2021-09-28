# MTSDataModel

MTSDataModel is a Python class that stores and manipulates economic multivariate time series data. Essentially it works as a wrapper on pandas library; it the core of MTSDataModel instanceobject is a pandas data frame that stores the data. Several data manipulation actions can be performed on the data frame.

## Installation

Conda enviroment for the use of MTSDataModel is created according to the instructions in [NoobQuant condaenv](https://github.com/NoobQuant/dsenvs/blob/main/condaenv.md). The specific installation command for *mts* environment is:

```
mamba create --name mts anaconda python=3.6.7 numpy=1.15.2 numpy-base=1.15.2 tzlocal=2.0.0 pandas=0.24.1 seaborn=0.11.0 rpy2==2.9.4 r=3.6.0 r-base=3.6.0 r-essentials=3.6.0 r-tidyverse=1.2.1 rtools=3.4.0 r-rjsdmx=2.1_0 r-seasonal=1.7.0 rstudio=1.1.456 r-wavelets=0.3_0.1 r-xlconnect=1.0.3
```

## Loading data into data model

Data is read in from a .csv file. This file needs to be in long format and contain following columns:
 - date column (string)
 - level 1 name column  for variable/feature, e.g. GDP (string)
 - level 2 name column for entity, e.g. country (string)
 - value column (numeric).

It is assumed for the input data that
 - each individual time series does not contain breaks
 - NAs can be present at start and end of individual time series. 

MTSDataModel initilizes to a data frame with two-leveled multi-index columns. First level is meant to represent variable names. Second level is meant to represent entity level (e.g. country, company etc.).

## Data manipulations

Several data manipulation operation can be performed on the initialized data frame within MTSDataModel object. These operations can be performed on different variables and/or entities.

### Variables and entities selection

Data manipulations are performed via class methods. For this level 1 names (variables) need be specified and passed into methods via list *variables*. When it comes to level 2 names (entities), following rules are followed:
 - **Implicit entities selection**: when list *entities* is left unspecified, then methods operate *only on entities for which all input variables are present*.
 - **Explicit entities selection**: when list *entities* is explicitly specified, then methods operate *only on variable/entity pairs (cross-product of the two input lists) specified*. If some variable/entity pair is not present in data an error will be thrown.

### Data pre-processing

Methods available for pre-processing of data are
 - DeflateVariables()
 - DetrendVariables()

### Feature engineering

Methods available for pre-processing of data are
 - MRADecomposition()
 - SumVariables()
 - ReduceVariableDimension()

