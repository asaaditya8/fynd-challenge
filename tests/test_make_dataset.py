from src.data import make_dataset
import os

def test_extract():
    assert make_dataset.extract_tar('data/raw/images.tar.gz', 'data/raw/') != 0

def test_process_img():
    make_dataset.process_image('data/external/65945.jpg', 'data/processed')
    assert os.path.isfile('data/processed/external/65945.jpg')