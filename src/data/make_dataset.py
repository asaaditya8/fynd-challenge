import tarfile
import argparse
import os

PATH = "data/raw/images.tar.gz"

def extract_tar(tar_path, out_path):
    if os.path.isdir(out_path):
        assert os.path.exists(tar_path)
        tarfile.open(tar_path).extractall(path=out_path)
        return 0

def main(args):
    extract_tar(args.tarInput, args.tarOutput)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract and process dataset')
    parser.add_argument('tarInput', type=str, help='path for tar file')
    parser.add_argument('tarOutput', type=str, help='path for output')

    args = parser.parse_args()

    main(args)

