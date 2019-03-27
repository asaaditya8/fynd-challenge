import fastai
from fastai.vision import *
from torchvision.transforms import transforms


def main():
    #inception_v3 requires the input size to be (299,299), whereas all of the other models expect (224,224).
    tfms = transforms.Compose([
        transforms.CenterCrop(224)
    ])

    path = os.path.abspath('data/raw/images/Train Directory/')
    testpath = os.path.abspath('data/raw/images/Test Directory/')
    fnames = [os.path.join(path, name) for path, subdirs, files in
              os.walk(os.path.abspath('data/raw/images/Train Directory')) for name in files]

    data = ImageDataBunch.from_name_func(path=path, fnames=fnames, label_func=lambda x: x.split('/')[-2],
                                         valid_pct=0.6, test=testpath, ds_tfms=(tfms, []))

    print(data.classes)

    ds = data.train_ds

    #visualise samples
    data.show_batch(rows=3, figsize=(5, 5))

if __name__ == '__main__':
    main()