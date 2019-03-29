import argparse
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from kerasonecycle.clr import OneCycleLR, LRFinder
from .model import create_model
from ..data.loader import get_data, BATCH_SIZE

MAX_LR = 0.1
total_samples = 1348
num_samples = ( total_samples//BATCH_SIZE ) * BATCH_SIZE

def find_lr(gen_dict, model, lr_dir):
    lr_callback = LRFinder(num_samples, batch_size=BATCH_SIZE,
                           minimum_lr=1e-3, maximum_lr=1e1,
                           lr_scale='exp', save_dir=lr_dir)

    model.layers[0].trainable = False
    model.layers[-1].trainable = False
    model.compile(SGD(), 'categorical_crossentropy', ['accuracy'])

    # Ensure that number of epochs = 1 when calling fit()
    model.fit_generator(gen_dict['train'], steps_per_epoch=total_samples//BATCH_SIZE, epochs=1,
                        batch_size=BATCH_SIZE, callbacks=[lr_callback])

    lr_callback.plot_schedule(clip_beginning=10, clip_endding=5)

def train(gen_dict, model, epochs):
    model.layers[0].trainable = False
    model.layers[-1].trainable = False
    model.compile(SGD(), 'categorical_crossentropy', ['accuracy'])

    lr_manager = OneCycleLR(MAX_LR)
    ckpt = ModelCheckpoint(save_best_only=True)
    logger = CSVLogger()

    model.fit_generator(gen_dict['train'], steps_per_epoch=total_samples // BATCH_SIZE, epochs=epochs,
                        batch_size=BATCH_SIZE, callbacks=[lr_manager, ckpt, logger])


def main(args):
    model = create_model()
    gen_dict = get_data(args.data_dir)

    if args.lr_dir:
        find_lr(gen_dict, model, args.lr_dir)

    if args.epochs:
        train(gen_dict, model, args.epochs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('data_dir', type=str, help='path for dataset')
    parser.add_argument('--lr_dir', type=str, help='path for lr result')
    parser.add_argument('--epochs', type=int, help='path for lr result')


    args = parser.parse_args()

    main(args)