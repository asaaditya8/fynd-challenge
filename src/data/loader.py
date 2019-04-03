from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg19 import preprocess_input
import os

BATCH_SIZE = 16

def get_data(inp_dir):
    var_list = ('train', 'valid', 'test')
    var_dict ={}
    for v in var_list:
        if v == 'train':
            var_dict[v] = ImageDataGenerator(rotation_range = 30,
                                             width_shift_range= 0.1,
                                             height_shift_range= 0.1,
                                             shear_range= 0.1,
                                             zoom_range= 0.1,
                                             horizontal_flip= True,
                                             fill_mode= "nearest",
                preprocessing_function=preprocess_input).flow_from_directory(
                                                  os.path.join(os.path.abspath(inp_dir), v),
                                                  batch_size=BATCH_SIZE,
                                                  target_size=(224,224))
        else:
            var_dict[v] = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(
                                                    os.path.join(os.path.abspath(inp_dir), v),
                                                    batch_size=BATCH_SIZE,
                                                    target_size=(224, 224))
    return var_dict