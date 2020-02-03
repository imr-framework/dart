import os

import matplotlib.pyplot as plt
import nibabel as nb

brats_path = '/Users/sravan953/Documents/CU/Projects/imr-framework/DART/Data/MICCAI_BraTS_2018_Data_Training'
subject = 'HGG/Brats18_CBICA_AOO_1'
path = os.path.join(brats_path, subject)
files = os.listdir(path)

i = 0
for f in files:
    if 'seg' not in f:
        temp_path = os.path.join(path, f)
        v = nb.load(temp_path)
        n = v.get_fdata()
        plt.subplot(2, 2, i + 1)
        plt.imshow(n[:, :, 77], cmap='gray')
        plt.axis('off')
        plt.title(f)
        i += 1

plt.tight_layout()
plt.show()
