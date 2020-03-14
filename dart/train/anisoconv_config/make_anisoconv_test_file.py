import os
from pathlib import Path

"""
Make test.txt containing <tumor type>/<subject ID>.
This test.txt config file is required for the anisotropic convolution model.
"""

PATH_TEST = Path(__file__).absolute().parent.parent / 'anisoconv_config' / 'test3.txt'
PATH_REGISTERED_BRATS = Path(
    r'C:\Users\sravan953\Documents\CU\Projects\imr-framework\DART\Data\4MICCAI_2020\DART_registered')

subjects = list(PATH_REGISTERED_BRATS.glob('*\*'))
subjects = list(
    map(lambda x: os.sep.join(str(x).split(os.sep)[-2:]), subjects))  # Split path to contain <tumor type>/<subject ID>
subjects = '\n'.join(subjects)  # Convert to string
PATH_TEST.write_text(subjects)
