# Udacity Machine Learning Engineering Nanodegree Capstone Project
## Author: Sam Strohkorb
## October 24, 2018
***
### Report
View the 'report.pdf' file to view the compiled report for this project. The report source file is 'report.tex'. All supporting files exist within the directory.

### Software
The software is split into four jupyter notebooks and two python file. The python files are used for visualization in one of the jupyter notebooks. The three jupyter notebooks are structured such that Web_get is at the bottom, Model is in the middle, and Udacity Capstone Project is at the top. This structure allows for better code organization. They use the 'ipynb' module to get one jupyter notebook to import another. By running the code in Udacity Capstone Project or Result Collection, the data from The Blue Alliance will automatically be downloaded and processed if it hasn't been already. This can be forced through the 'update' function in Model. The fourth jupyter notebook is the Result Collection. Running this will generate all of the results from the models.

### Required Libraries
Python 3.4 or later

ipynb, matplotlib, sklearn, tqdm, numpy, pandas, xgboost, urllib3, json, os, and warnings

The most recent versions of these modules will work
