import os

import textract
import pandas as pd

# Construct ID mappings to tumour type
# For example, HGG: [1, 4, 5...]; LGG: [103, 107, 141...
hgg_path = '/Users/sravan953/Documents/CU/Projects/imr-framework/DART/Data/MICCAI_BraTS_2018_Data_Training/HGG'
lgg_path = '/Users/sravan953/Documents/CU/Projects/imr-framework/DART/Data/MICCAI_BraTS_2018_Data_Training/LGG'
tumour_type_dict = {'HGG': [x.split('_')[-2] for x in os.listdir(hgg_path)],
                    'LGG': [x.split('_')[-2] for x in os.listdir(lgg_path)]}

docx_path = '/Users/sravan953/Documents/CU/Projects/imr-framework/DART/Data/Girish_reports/docx'
columns = ['Subject', 'Type', 'Lesion', 'T1', 'T1ce', 'T2', 'FLAIR', 'Edema', 'Comments']
all_features = []
for docx_file in os.listdir(docx_path):
    docx_text = textract.process(os.path.join(docx_path, docx_file)).decode()
    features = [None] * len(columns)  # Columns of feature vector

    file_id = docx_file.replace('.docx', '').replace('brats_', '')
    features[0] = file_id
    features[1] = 'HGG' if file_id in tumour_type_dict['HGG'] else 'LGG'

    all_lines = docx_text.split('\n')
    for i, line in enumerate(all_lines):
        line = line.strip()
        if line == '':
            continue
        elif 'Lesion:' in line:
            comment = line.split(':')[1].strip()
            id = columns.index('Lesion')
            features[id] = comment
        elif 'T1:' in line:
            comment = line.split(':')[1].strip()
            id = columns.index('T1')
            features[id] = comment
        elif 'T1+contrast:' in line:
            comment = line.split(':')[1].strip()
            id = columns.index('T1ce')
            features[id] = comment
        elif ('T2/Flair:' in line) or ('T2 / Flair:' in line):
            comment = line.split(':')[1].strip()
            id = columns.index('T2')
            features[id] = comment
            id = columns.index('FLAIR')
            features[id] = comment
        elif 'Edema:' in line:
            comment = line.split(':')[1].strip()
            id = columns.index('Edema')
            features[id] = comment
        elif 'Comments' in line:
            comment = []
            # If we encounter the 'Comments' tag, we want to extract the following lines
            # Usually, 'Comments' is toward the ending of the file
            # So, we extract lines till EOF
            for j in all_lines[i + 1:]:
                if j != '\n' and j != '':
                    comment.append(j.strip())
            id = columns.index('Comments')
            features[id] = comment

    all_features.append(features)

df = pd.DataFrame(data=all_features, columns=columns)
df.to_excel('/Users/sravan953/Documents/CU/Projects/imr-framework/DART/Data/Girish_reports/reports.xlsx')
