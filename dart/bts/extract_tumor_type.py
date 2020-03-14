import os
from pathlib import Path

import pandas as pd

"""
Read the report spreadsheet and make a pandas.Series of matching tumor types.
Required when the radiologist does not fill out the tumor type column while annotating BraTS 2018 data.
"""

PATH_REPORT = r'../train/REPORT_ZENAS2.xlsx'  # <-- CHANGE ME
PATH_MICCAI_data = r'C:\Users\sravan953\Documents\CU\Projects\imr-framework\DART\Data\MICCAI_BraTS_2018_Data_Training'

df = pd.read_excel(PATH_REPORT)
arr_reported_subject_idx = df['Subject ID']

arr_files = list(Path(PATH_MICCAI_data).glob('*/*'))  # List of all files in the BraTS 2018 dataset
arr_subject_idx = list(
    map(lambda filename: str(filename).split('_')[-2], arr_files))  # Extract subject ID from arr_files
arr_tumor_type = list(
    map(lambda filename: str(filename).split(os.sep)[-2], arr_files))  # Extract tumor type from arr_files

reported_tumor_types = []
for reported_subject_id in arr_reported_subject_idx:
    tumor_type = arr_tumor_type[arr_subject_idx.index(str(reported_subject_id))]
    reported_tumor_types.append(tumor_type)

series = pd.Series(reported_tumor_types)
# series.to_excel('TUMOR_TYPES.xlsx')
