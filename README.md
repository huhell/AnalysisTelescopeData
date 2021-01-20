# Analysis of space telescope data

Purpose: mining, cleaning, and basic analysis of space telescope data

Author: Hugo Hellard

## Summary
I use public raw data from the Kepler space telescope, which observed the transit of the exoplanet HAT-P-7b in front of its host star.
The data can be found and downloaded at: https://archive.stsci.edu/kepler/data_search/search.php

I show here how to extract, clean, and eventually analyze the data to estimate basic planetary parameters. The targeted data is a time series of stellar flux 
measurements as the planet passes in front of its star. The procedure thus involves data analysis and modeling.

## Files and requirements
There are two Python scripts:
- `read_files.py` which extracts, cleans, normalize and bin the data.
- `model_data.py` which analyzes the previously binned data to estimate basic planetary parameters. This analysis uses Bayesian statistics, and particularly 
a Markov Chain Monte Carlo algorithm. This script requires more scpecific Python libraries (`PyAstronomy`, and `mc3`).

Each script requires some Python libraries:
- For `read_files.py`: `astropy`, `scipy`, `numpy`, `pandas`, and `matplotlib`
- For `model_data.py`: `numpy`, `pandas`, `matplotlib`, `PyAstronomy`, `mc3`

## Usage
Once you have the required Python libraries, simply download the files. Then, simply run first the `read_files.py` script. And, if you wish, then you may run 
the `model_data.py` script.

The result of the analysis, and in particular the unusual shape of the residual curve, shows that further analysis of the data is needed.
