#!/usr/bin/env python3
import os
import sys
import urllib.request
import subprocess

WEIGHTS_DIR = 'weights'
os.makedirs(WEIGHTS_DIR, exist_ok=True)

SAM_URL = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'
SAM_PATH = os.path.join(WEIGHTS_DIR, 'sam_vit_h_4b8939.pth')

MEDSAM_GDRIVE_ID = '1UAmWL88roYR7wKlnApw5Bcuzf2iQgk6_'
MEDSAM_PATH = os.path.join(WEIGHTS_DIR, 'medsam_vit_b.pth')


def _progress(count, block, total):
    pct = min(count * block * 100 // total, 100)
    print(f'\r  {pct}%', end='', flush=True)


def download_sam():
    if os.path.exists(SAM_PATH):
        print(f'[skip] SAM weights already at {SAM_PATH}')
        return
    print(f'Downloading SAM ViT-H → {SAM_PATH}')
    urllib.request.urlretrieve(SAM_URL, SAM_PATH, reporthook=_progress)
    print(f'\n[done] {SAM_PATH}')


def download_medsam():
    if os.path.exists(MEDSAM_PATH):
        print(f'[skip] MedSAM weights already at {MEDSAM_PATH}')
        return
    try:
        import gdown
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'gdown', '-q'])
        import gdown
    print(f'Downloading MedSAM ViT-B → {MEDSAM_PATH}')
    gdown.download(id=MEDSAM_GDRIVE_ID, output=MEDSAM_PATH, quiet=False)
    if not os.path.exists(MEDSAM_PATH):
        print('\n[error] gdown failed. Download manually from:')
        print('  https://github.com/bowang-lab/MedSAM')
        print(f'  and save to {MEDSAM_PATH}')
        sys.exit(1)
    print(f'[done] {MEDSAM_PATH}')


if __name__ == '__main__':
    download_sam()
    download_medsam()
