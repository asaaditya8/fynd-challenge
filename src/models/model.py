import keras
from keras.applications.vgg19 import VGG19
from keras import layers
from keras.models import Model, Sequential

def create_top(num_classes:int):
    """
    Creates fastai style top
    :param num_classes: Number of classes
    :return: Top model
    """
    top = Sequential([
        layers.BatchNormalization(),
        layers.Dense(512, use_bias=False, input_shape=(64+128+256+512+512,)),
        layers.Activation('relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(512, use_bias=False),
        layers.Activation('relu'),
        layers.BatchNormalization(),
        layers.Dense(num_classes, activation='softmax', use_bias=False)
        ])
    return top


def create_model(weights = 'imagenet'):
    base = VGG19(include_top=False, weights=weights, pooling=None)
    top = create_top(12)

    # I think average pooling should be done on conv layer just before max pooling layer,
    # because why 2 pooling layers should be there?
    indices = [2, 5, 10, 15, 20]
    intermediate_features = [layers.GlobalAveragePooling2D()(base.get_layer(index=i).output) for i in indices ]

    x = layers.Concatenate()(intermediate_features)
    x = top(x)

    return Model(base.input, x)


def main():
    pass

if __name__ == '__main__':
    main()