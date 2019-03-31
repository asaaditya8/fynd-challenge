import sys
sys.path.append("..")

import argparse
import os
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from model import create_model
from src.data.make_dataset import TEST_SIZE, N_VAL
from src.data.loader import get_data, BATCH_SIZE
from src.kerasonecycle.clr import OneCycleLR, LRFinder

PATIENCE = 4
MAX_LR = 0.032
N_SAMPLES = int(3370 * (1 - TEST_SIZE))
num_samples = (N_SAMPLES // BATCH_SIZE) * BATCH_SIZE

class Learner:
    def __init__(self, gen_dict, model, trainable:bool = False):
        self.gen_dict = gen_dict
        self.model = model
        self.set_base_trainable(trainable)
        print(self.model.summary())

    def set_base_trainable(self, trainable:bool = False):
        self.model.layers[1].trainable = trainable
        self.model.compile(SGD(), 'categorical_crossentropy', ['accuracy'])

    def find_lr(self, lr_dir):
        """
        This creates a plot to find MAX_LR
        :param lr_dir: directory to save plot
        :return: None
        """
        lr_callback = LRFinder(num_samples, batch_size=BATCH_SIZE,
                               minimum_lr=1e-3, maximum_lr=1e1,
                               lr_scale='exp', save_dir=os.path.abspath(lr_dir))

        # Ensure that number of epochs = 1 when calling fit()
        self.model.fit_generator(self.gen_dict['train'], steps_per_epoch=N_SAMPLES // BATCH_SIZE, epochs=1,
                                 callbacks=[lr_callback])

        lr_callback.plot_schedule(clip_beginning=10, clip_endding=5)

        # use this to plot manually if the above fails
        # LRFinder.plot_schedule_from_file('../reports/figures', clip_beginning=10, clip_endding=5)

    def fit(self, ckpt_path, log_path, epochs=10):
        """
        Model versioning will be done by dvc so provide a trivial name for checkpoint and log
        :param epochs: Number of epochs to train 
        :param ckpt_path: File path including file name for saving model 
        :param log_path: File path including file name for logging training history
        :return: None
        """
        #Set MAX_LR by running find_lr
        lr_manager = OneCycleLR(num_samples, epochs, BATCH_SIZE, MAX_LR)
        es = EarlyStopping(patience=PATIENCE)
        ckpt = ModelCheckpoint(os.path.abspath(ckpt_path), save_best_only=True)
        logger = CSVLogger(os.path.abspath(log_path))

        self.model.fit_generator(self.gen_dict['train'], steps_per_epoch=N_SAMPLES // BATCH_SIZE, epochs=1,
                                 callbacks=[lr_manager, es, ckpt, logger],
                                 validation_data=self.gen_dict['valid'],
                                 validation_steps=N_VAL // BATCH_SIZE)

        self.set_base_trainable(True)

        self.model.fit_generator(self.gen_dict['train'], steps_per_epoch=N_SAMPLES // BATCH_SIZE, epochs=epochs,
                                 callbacks=[lr_manager, es, ckpt, logger],
                                 validation_data=self.gen_dict['valid'],
                                 validation_steps=N_VAL // BATCH_SIZE)


def main(args):
    model = create_model()
    gen_dict = get_data(args.data_dir)
    learner = Learner(gen_dict, model)

    if args.lr_dir:
        learner.find_lr(args.lr_dir)

    if args.epochs:
        assert args.ckpt_path is not None
        assert args.log_path is not None
        learner.fit(args.ckpt_path, args.log_path, args.epochs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('data_dir', type=str, help='path for dataset')
    parser.add_argument('--lr_dir', type=str, help='path for lr result')
    parser.add_argument('--epochs', type=int, help='number of epochs to train')
    parser.add_argument('--ckpt_path', type=str, help='path for saving model')
    parser.add_argument('--log_path', type=str, help='path for saving training log')

    args = parser.parse_args()

    main(args)