Bayesian optimization with search space prescreening via outlier detection (ODBO)
## Overview
This repository includes the codes and results for our paper:
***ODBO: Bayesian Optimization with Search Space Prescreening for Directed Protein Evolution***

ODBO is written as a **maximization** algorithm to search the best experimental design with desired properties. The initial sample generators and different encodings are also included in this repo

## Installation
Please first clone our repo and install using the setup.py. All the dependencies are listed in the ```requirements.txt```.

```
git@github.com:sherrylixuecheng/ODBO.git
cd ODBO
pip install requirements.txt (if needed)
python setup.py install 
```

## Content list
The descriptions of files in each folder are listed in the corresponding ```README.md``` file in the folder 

[datasets](datasets) contains the raw data for four protein datasets obtained from the original publications and \url{https://github.com/gitter-lab/nn4dms}
[examples]