import re
from pathlib import Path


def main():
    """
     Map neuroanatomy labels in `freesurfer_labels.txt` to corresponding IDs.
     Both labels and IDs are contained in the text file.
    """
    path_freesurfer_labels = Path(__file__).parent.parent / 'talk' / 'freesurfer_labels.txt'
    freesurfer_labels = path_freesurfer_labels.read_text().split('\n')
    freesurfer_labels = [re.split(r'\s+', x) for x in freesurfer_labels]
    freesurfer_idx = [int(x[0]) for x in freesurfer_labels if x[0] != '']
    freesurfer_regions = [x[1] for x in freesurfer_labels if x[0] != '']
    freesurfer_labels = dict(zip(freesurfer_idx, freesurfer_regions))
    return freesurfer_labels
