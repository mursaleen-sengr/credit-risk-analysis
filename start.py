
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
import warnings
import os




# Load the dataset
a1 = pd.read_excel("C:\\xxx\\Desktop\\CampusX_10_Apr\\Credit_risk_excel_files\\case_study1.xlsx")
a2 = pd.read_excel("C:\\xxx\\Desktop\\CampusX_10_Apr\\Credit_risk_excel_files\\case_study2.xlsx")