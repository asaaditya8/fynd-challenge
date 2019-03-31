import keras
from keras.applications.xception import Xception, preprocess_input
from keras import layers
from keras.models import Model, Sequential

def create_top(num_classes:int):
    """
    Creates fastai style top
    :param num_classes: Number of classes
    :return: Top model
    """
    top = Sequential([
        layers.Dropout(0.5, input_shape=(2048*2,)),
        layers.Dense(512, use_bias=False),
        layers.Activation('relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(512, use_bias=False),
        layers.Activation('relu'),
        layers.Dense(num_classes, activation='softmax', use_bias=False)
        ])
    return top


def create_model(weights = 'imagenet'):
    base = Xception(include_top=False, weights=weights, pooling=None)
    top = create_top(12)
    inp = layers.Input((None, None, 3))
    x = base(inp)
    x = layers.Concatenate()([layers.GlobalAveragePooling2D()(x), layers.GlobalMaxPool2D()(x)])
    x = top(x)

    return Model(inp, x)


def main():
    pass

if __name__ == '__main__':
    main()