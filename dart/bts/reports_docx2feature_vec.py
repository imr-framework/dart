import os
import pickle

import textract

# Construct ID mappings to tumour type
# For example, HGG: [1, 4, 5...]; LGG: [103, 107, 141...
hgg_path = '/Users/sravan953/Documents/CU/Projects/imr-framework/DART/Data/MICCAI_BraTS_2018_Data_Training/HGG'
lgg_path = '/Users/sravan953/Documents/CU/Projects/imr-framework/DART/Data/MICCAI_BraTS_2018_Data_Training/LGG'
subject_name_tumour_type_dict = {'HGG': sorted(os.listdir(hgg_path)), 'LGG': sorted(os.listdir(lgg_path))}
subject_id_tumour_type_dict = {'HGG': [x.split('_')[-2] for x in subject_name_tumour_type_dict['HGG']],
                               'LGG': [x.split('_')[-2] for x in subject_name_tumour_type_dict['LGG']]}

docx_path = '/Users/sravan953/Documents/CU/Projects/imr-framework/DART/Data/Girish_reports/docx'
subject_name = []
tumour_type = []
edema = []
mass_effect = []

for docx_file in os.listdir(docx_path):
    docx_text = textract.process(os.path.join(docx_path, docx_file)).decode()
    file_id = docx_file.replace('.docx', '').replace('brats_', '')
    if file_id in subject_id_tumour_type_dict['HGG']:
        subject_name.append(subject_name_tumour_type_dict['HGG'][subject_id_tumour_type_dict['HGG'].index(file_id)])
    else:
        subject_name.append(subject_name_tumour_type_dict['LGG'][subject_id_tumour_type_dict['LGG'].index(file_id)])
    tumour_type.append(1 if file_id in subject_id_tumour_type_dict['HGG'] else 0)

    all_lines = docx_text.split('\n')
    for i, line in enumerate(all_lines):
        line = line.strip()
        if 'Edema:' in line:
            comment = line.split(':')[1].strip()
            mass_effect.append(1 if 'mass effect' in comment and 'no mass effect' not in comment else 0)
            if 'mild' in comment or 'Mild' in comment:
                edema.append(1)
            elif 'moderate' in comment or 'Moderate' in comment:
                edema.append(2)
            elif 'extensive' in comment or 'Extensive' in comment:
                edema.append(3)
            else:
                edema.append(0)

feature_vector = {'Subject': subject_name, 'Tumour type': tumour_type, 'Edema': edema, 'Mass effect': mass_effect}
save_path = '/Users/sravan953/Documents/CU/Projects/imr-framework/DART/Data/Girish_reports'
with open(os.path.join(save_path, 'report_feature_vector.p'), 'wb') as p:
    pickle.dump(feature_vector, p)
