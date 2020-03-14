from pathlib import Path

from train.anisoconv_config.anisoconv_seg import test

PATH_CONFIG_FILE = Path(__file__).absolute().parent.parent / 'anisoconv_config' / 'config.txt'
test(str(PATH_CONFIG_FILE))
