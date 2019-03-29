from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input
import os

BATCH_SIZE = 32

def get_data(inp_dir):
    aug = ImageDataGenerator(preprocessing_function=preprocess_input)
    var_list = ('train', 'valid', 'test')
    var_dict ={}
    for v in var_list:
        var_dict[v] = aug.flow_from_directory(os.path.join(os.path.abspath(inp_dir), v),
                                              batch_size=BATCH_SIZE,
                                              target_size=(224,224))
    return var_dict