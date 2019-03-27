import tarfile
import argparse
import os

PATH = "data/raw/images.tar.gz"

def extract_tar(tar_path, out_path):
    tar_path, out_path = map(os.path.abspath, (tar_path, out_path))

    if not os.path.isdir(os.path.join(out_path, 'images')):
        assert os.path.isfile(tar_path)
        tarfile.open(tar_path).extractall(path=out_path)
        os.rmdir(os.path.join(out_path, 'images', 'Train Directory', 'Predicted'))
        return 0

def main(args):
    extract_tar(args.tarInput, args.tarOutput)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract and process dataset')
    parser.add_argument('tarInput', type=str, help='path for tar file')
    parser.add_argument('tarOutput', type=str, help='path for output')

    args = parser.parse_args()

    main(args)

