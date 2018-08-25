
'''
 * @author [Zizhao Zhang]
 * @email [zizhao@cise.ufl.edu]
 * @create date 2017-05-19 03:06:43
 * @modify date 2017-05-19 03:06:43
 * @desc [description]
'''

from data_generator.image import ImageDataGenerator#, ClassDataGenerator
import scipy.misc as misc
import numpy as np
import os, glob, itertools
from PIL import ImageFile
from PIL import Image as pil_image
ImageFile.LOAD_TRUNCATED_IMAGES = True
from matplotlib import pyplot as plt

# Modify this for data normalization 
def preprocess(img, mean, std, label, normalize_label=True):
    HOUNSFIELD_LIMITS = (0, 255)

    out_img = img.copy()
    out_img[(out_img < HOUNSFIELD_LIMITS[0])] = HOUNSFIELD_LIMITS[0]
    out_img[(out_img > HOUNSFIELD_LIMITS[1])] = HOUNSFIELD_LIMITS[1]
    out_img -= HOUNSFIELD_LIMITS[0]
    out_img = out_img / (HOUNSFIELD_LIMITS[1] - HOUNSFIELD_LIMITS[0])
    out_img = (2*out_img) - 1
    # this is for _equalized
    # out_img = img / img.max() # scale to [0,1]
    # out_img = (out_img - np.array(mean)) / np.array(std)
    if len(label.shape) == 4:
        label = label[:,:,:,1]

    # if normalize_label:
    #     if np.unique(label).size > 2:
    #         print ('WRANING: the label has more than 2 classes. Set normalize_label to False')
    #     label = label / label.max() # if the loaded label is binary has only [0,255], then we normalize it
    # print(label[label>0])
    # import numpngw
    # print(img.shape)
    # numpngw.write_png('/tmp/blat_{0}.png'.format(str(np.random.randint(1000))), (img.squeeze()[1]).astype('uint8'))
    # try:
    #     if label[label>0].any():
    #         numpngw.write_png('/tmp/blat{0}.png'.format(str(np.random.randint(1000))), (label.squeeze()*200).astype('uint8'))
    # except:
    #     pass
    label = (label > 0).astype(np.int32)

    return out_img, label

def deprocess(img, mean, std, label):
    out_img = (img + 1)/2
    # out_img = img / img.max() # scale to [0,1]
    # out_img = (out_img * np.array(std).reshape(1,1,3)) + np.array(std).reshape(1,1,3)
    if out_img.shape[-1] == 3:  # take middle channel :(
        out_img = out_img[:, :, 1]
    out_img = out_img * 255.0

    return out_img.astype(np.uint8), label.astype(np.uint8)

'''
    Use the Keras data generators to load train and test
    Image and label are in structure:
        train/
            img/
                0/
            gt/
                0/

        test/
            img/
                0/
            gt/
                0/

'''
def dataLoader(path, batch_size, imSize, train_mode=True, mean=0.5, std=0.5):
    # train_mode=True
    # image normalization default: scale to [-1,1]
    def imerge(a, b):
        for img, label in itertools.zip_longest(a,b):
            # j is the mask: 1) gray-scale and int8
            img, label = preprocess(img, mean, std, label, normalize_label=False)
            # print('MERGE')
            # print(label.max())
            # print(img.max())
            # print(img.min())
            yield img, label
    
    # augmentation parms for the train generator
    if train_mode:
        train_data_gen_args = dict(
                        horizontal_flip=True,
                        # vertical_flip=True,
                        vertical_flip=True,
                        zoom_range=0.05,
                        elastic_deformation_params=None,
                        # elastic_deformation_params=[10, 5],
                        # elastic_deformation_params=[100, 1],
                        rotation_range=180,
                        # rotation_range=205,
                        width_shift_range=0.01,
                        height_shift_range=0.01,
                        fill_mode='constant',
                        image_size=imSize
                        )
    else:
        train_data_gen_args = dict()

    color_mode = 'rgb' if imSize[-1] == 3 else 'grayscale'

    # seed has to been set to synchronize img and mask generators
    seed = 1
    train_image_datagen = ImageDataGenerator(**train_data_gen_args).flow_from_directory(
                                path+'img',
                                class_mode=None,
                                target_size=imSize[:2],
                                batch_size=batch_size,
                                color_mode=color_mode,
                                # save_to_dir='augmented',
                                save_prefix='img',
                                seed=seed,
                                shuffle=train_mode)
    train_mask_datagen = ImageDataGenerator(**train_data_gen_args).flow_from_directory(
                                path+'gt',
                                class_mode=None,
                                target_size=imSize[:2],
                                batch_size=batch_size,
                                color_mode=color_mode,  # mask is always grayscale
                                # save_to_dir='augmented',
                                save_prefix='mask',
                                seed=seed,
                                shuffle=train_mode)
                                
    samples = train_image_datagen.samples
    generator = imerge(train_image_datagen, train_mask_datagen)

    return generator, samples

# 
# def dataLoaderClassification(path, batch_size, imSize, classes, train_mode=True):
#     # augmentation parms for the train generator
#     if train_mode:
#         train_data_gen_args = dict(
#             horizontal_flip=False,
#             # vertical_flip=True,
#             vertical_flip=False,
#             zoom_range=0.15,
#             # elastic_deformation_params=[10, 5],
#             # elastic_deformation_params=[100, 1],
#             rotation_range=10,
#             # rotation_range=205,
#             width_shift_range=0.1,  # TODO: good?
#             height_shift_range=0.01,  # TODO: good?
#             fill_mode='constant',
#             image_size=imSize
#         )
#     else:
#         train_data_gen_args = dict()
# 
#     color_mode = 'rgb' if imSize[-1] == 3 else 'grayscale'
# 
#     # seed has to been set to synchronize img and mask generators
#     seed = 1
#     np.random.seed(seed)
#     generator = ClassDataGenerator(
#         file_name=path,
#         batch_size=batch_size,
#         # data_split=100,
#         # start=0,
#         # end=None,
#         root_name_x='img',
#         root_name_y='grade',
#         is_train=train_mode,
#         crop_size=(100, 200),  # TODO: good?
#         classes=classes,
#         imgen_params=train_data_gen_args
#     )
#         #     path+'img',
#         #     class_mode=None,
#         #     target_size=imSize[:2],
#         #     batch_size=batch_size,
#         #     color_mode=color_mode,
#         #     # save_to_dir='augmented',
#         #     save_prefix='img',
#         #     seed=seed,
#         #     shuffle=train_mode)
# 
#     # train_image_datagen = ImageDataGenerator(**train_data_gen_args).flow_from_directory(
#     #     path+'img',
#     #     class_mode=None,
#     #     target_size=imSize[:2],
#     #     batch_size=batch_size,
#     #     color_mode=color_mode,
#     #     # save_to_dir='augmented',
#     #     save_prefix='img',
#     #     seed=seed,
#     #     shuffle=train_mode)
#     # train_mask_datagen = ImageDataGenerator(**train_data_gen_args).flow_from_directory(
#     #     path+'gt',
#     #     class_mode=None,
#     #     target_size=imSize[:2],
#     #     batch_size=batch_size,
#     #     color_mode='grayscale',  # mask is always grayscale
#     #     # save_to_dir='augmented',
#     #     save_prefix='mask',
#     #     seed=seed,
#     #     shuffle=train_mode)
# 
#     samples = generator.total_len
# 
#     return generator, samples
# 
