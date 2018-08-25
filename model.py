'''
 * @author [Zizhao Zhang]
 * @email [zizhao@cise.ufl.edu]
 * @create date 2017-05-25 02:21:13
 * @modify date 2017-05-25 02:21:13
 * @desc [description]
'''
import tensorflow as tf
import keras
from keras import models
from keras import layers


class UNet():
    def __init__(self):
        print ('build UNet ...')

    def get_crop_shape(self, target, refer):
        # width, the 3rd dimension
        cw = (target.get_shape()[2] - refer.get_shape()[2]).value
        assert (cw >= 0)
        if cw % 2 != 0:
            cw1, cw2 = int(cw/2), int(cw/2) + 1
        else:
            cw1, cw2 = int(cw/2), int(cw/2)
        # height, the 2nd dimension
        ch = (target.get_shape()[1] - refer.get_shape()[1]).value
        assert (ch >= 0)
        if ch % 2 != 0:
            ch1, ch2 = int(ch/2), int(ch/2) + 1
        else:
            ch1, ch2 = int(ch/2), int(ch/2)

        return (ch1, ch2), (cw1, cw2)

    def create_model(self, img_shape, num_class):
        concat_axis = 3

        shape = img_shape + [1] if len(img_shape) == 1 else img_shape
        inputs = layers.Input(shape=shape)

        conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_1')(inputs)
        conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
        pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
        pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
        conv5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

        up_conv5 = layers.UpSampling2D(size=(2, 2))(conv5)
        ch, cw = self.get_crop_shape(conv4, up_conv5)
        crop_conv4 = layers.Cropping2D(cropping=(ch,cw))(conv4)
        up6 = layers.concatenate([up_conv5, crop_conv4], axis=concat_axis)
        conv6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
        conv6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

        up_conv6 = layers.UpSampling2D(size=(2, 2))(conv6)
        ch, cw = self.get_crop_shape(conv3, up_conv6)
        crop_conv3 = layers.Cropping2D(cropping=(ch,cw))(conv3)
        up7 = layers.concatenate([up_conv6, crop_conv3], axis=concat_axis) 
        conv7 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
        conv7 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

        up_conv7 = layers.UpSampling2D(size=(2, 2))(conv7)
        ch, cw = self.get_crop_shape(conv2, up_conv7)
        crop_conv2 = layers.Cropping2D(cropping=(ch,cw))(conv2)
        up8 = layers.concatenate([up_conv7, crop_conv2], axis=concat_axis)
        conv8 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
        conv8 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

        up_conv8 = layers.UpSampling2D(size=(2, 2))(conv8)
        ch, cw = self.get_crop_shape(conv1, up_conv8)
        crop_conv1 = layers.Cropping2D(cropping=(ch,cw))(conv1)
        up9 = layers.concatenate([up_conv8, crop_conv1], axis=concat_axis)
        conv9 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
        conv9 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

        ch, cw = self.get_crop_shape(inputs, conv9)
        conv9 = layers.ZeroPadding2D(padding=((ch[0], ch[1]), (cw[0], cw[1])))(conv9)
        # conv10 = layers.Conv2D(num_class, (1, 1))(conv9)
        conv10 = layers.Conv2D(1, (1, 1), activation='sigmoid')(conv9)

        model = models.Model(inputs=inputs, outputs=conv10)

        return model

class ClassicNet():
    def __init__(self):
        print ('build Classic net ...')

    def create_model(self, img_shape, num_class, dropout_rate=None):
        '''leave dropout_rate=None for batchnorm'''
        model = models.Sequential([
            layers.Conv2D(48, (4, 4), activation='relu', padding='same', input_shape=img_shape),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(dropout_rate/2) if dropout_rate is not None else layers.BatchNormalization(),
            layers.Conv2D(48, (5, 5), activation='relu', padding='same'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(dropout_rate/2) if dropout_rate is not None else layers.BatchNormalization(),
            layers.Conv2D(48, (5, 5), activation='relu', padding='same'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(dropout_rate/2) if dropout_rate is not None else layers.BatchNormalization(),
            layers.Flatten(),
            layers.Dense(200, activation='relu'),
            layers.Dropout(dropout_rate) if dropout_rate is not None else layers.BatchNormalization(),  # FIXME: also batchnorm(?)
            layers.Dense(num_class, activation='softmax')
        ])
        # inputs = layers.Input(shape=img_shape)
        # conv1 = layers.Conv2D(48, (4, 4), activation='relu', padding='same', name='conv1_1')(inputs)
        # pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
        # conv2 = layers.Conv2D(48, (5, 5), activation='relu', padding='same')(pool1)
        # pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
        # conv3 = layers.Conv2D(48, (5, 5), activation='relu', padding='same')(pool2)
        # pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
        # fc1 = layers.Dense(200, activation='relu')(pool3)
        # dropout1 = layers.Dropout(dropout_rate)(fc1, training=True)
        # fc2 = layers.Dense(num_class)(dropout1)
        # model = models.Model(inputs=inputs, outputs=fc2)

        return model

    #
    #
    # x, is_train = augment(x)
    #     h_conv1, W_conv1 = cu.conv2dpool(x, [4, 4, 1, 48], use_bias=True)
    #     h_conv2, W_conv2 = cu.conv2dpool(h_conv1, [5, 5, 48, 48], use_bias=True)
    #     h_conv3, W_conv3 = cu.conv2dpool(h_conv2, [5, 5, 48, 48], use_bias=True)
    #     h_fc1, W_fc1 = cu.linear(h_conv3, 200, activation=tf.nn.relu)
    #     keep_prob = tf.cond(is_train, lambda: tf.constant(dropout_keep_prob, dtype=tf.float32), lambda: tf.constant(1.0, dtype=tf.float32))
    #     h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    #     y_conv, W_fc2 = cu.linear(h_fc1_drop, n_labels)


class AdiNet():
    def __init__(self):
        print ('build Adi net ...')

    def create_model(self, img_shape, num_class, dropout_rate=0.25):
        model = keras.Sequential()

        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=img_shape,strides=1))
        model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(dropout_rate))

        model.add(layers.Conv2D(64, (3, 3), activation='relu',strides=1))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(dropout_rate))

        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(1, 3)))
        model.add(layers.Dropout(dropout_rate))

        model.add(layers.Conv2D(64, (1, 2), activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 1)))
        model.add(layers.Dropout(dropout_rate))

        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dropout(0.5))

        model.add(layers.Dense(num_class, activation='softmax'))





class creat_deep_Unet():
    def __init__(self):
        print ('build deep Unet ...')


    def get_crop_shape(self, target, refer):
        # width, the 3rd dimension
        cw = (target.get_shape()[2] - refer.get_shape()[2]).value
        assert (cw >= 0)
        if cw % 2 != 0:
            cw1, cw2 = int(cw/2), int(cw/2) + 1
        else:
            cw1, cw2 = int(cw/2), int(cw/2)
        # height, the 2nd dimension
        ch = (target.get_shape()[1] - refer.get_shape()[1]).value
        assert (ch >= 0)
        if ch % 2 != 0:
            ch1, ch2 = int(ch/2), int(ch/2) + 1
        else:
            ch1, ch2 = int(ch/2), int(ch/2)

        return (ch1, ch2), (cw1, cw2)

    def create_model(self, img_shape=(512, 512, 3), num_classes=1, dropout_rate=None):

        inputs = layers.Input(shape=img_shape)
        # 1024

        down0 = layers.Conv2D(64, (3, 3), padding='same')(inputs)
        down0 = layers.Activation('relu')(down0)
        down0 = layers.Conv2D(64, (3, 3), padding='same')(inputs)
        down0 = layers.Activation('relu')(down0)
        plus_0 = layers.Conv2D(32, (2, 2), padding='same')(down0)
        down0 = layers.MaxPooling2D((2, 2), strides=(2, 2))(down0)
        down0 = layers.Activation('relu')(down0)

        # 512

        down1 = layers.Conv2D(64, (3, 3), padding='same')(down0)
        down1 = layers.Activation('relu')(down1)
        down1 = layers.Conv2D(32, (3, 3), padding='same')(down1)
        plus_1 = layers.concatenate([down1, down0], axis=3)
        down1= layers.MaxPooling2D((2, 2), strides=(2, 2))(plus_1)
        down1 = layers.Activation('relu')(down1)

        # 256

        down2 = layers.Conv2D(64, (3, 3), padding='same')(down1)
        down2 = layers.Activation('relu')(down2)
        down2 = layers.Conv2D(32, (2, 2), padding='same')(down2)
        plus_2 = layers.concatenate([down2, down1], axis=3)
        down2 = layers.MaxPooling2D((2, 2), strides=(2, 2))(plus_2)
        down2 = layers.Activation('relu')(down2)

        # 128

        down3 = layers.Conv2D(64, (3, 3), padding='same')(down2)
        down3 = layers.Activation('relu')(down3)
        down3 = layers.Conv2D(32, (2, 2), padding='same')(down3)
        plus_3 =layers.concatenate([down3, down2], axis=3)
        down3 = layers.MaxPooling2D((2, 2), strides=(2, 2))(plus_3)
        down3 = layers.Activation('relu')(down3)

        # 64

        down4 = layers.Conv2D(64, (3, 3), padding='same')(down3)
        down4 = layers.Activation('relu')(down4)
        down4 = layers.Conv2D(32, (2, 2), padding='same')(down4)
        plus_4 = layers.concatenate([down4, down3], axis=3)
        down4 = layers.MaxPooling2D((2, 2), strides=(2, 2))(plus_4)
        down4 = layers.Activation('relu')(down4)

        # 32

        down5 = layers.Conv2D(64, (3, 3), padding='same')(down4)
        down5 = layers.Activation('relu')(down5)
        down5 = layers.Conv2D(32, (2, 2), padding='same')(down5)
        plus_5 = layers.concatenate([down5, down4], axis=3)
        down5 = layers.MaxPooling2D((2, 2), strides=(2, 2))(plus_5)
        down5 = layers.Activation('relu')(down5)

        # 16

        down6 = layers.Conv2D(64, (3, 3), padding='same')(down5)
        down6 = layers.Activation('relu')(down6)
        down6 = layers.Conv2D(32, (2, 2), padding='same')(down6)
        plus_6 = layers.concatenate([down6, down5], axis=3)
        down6 = layers.MaxPooling2D((2, 2), strides=(2, 2))(plus_6)
        down6 = layers.Activation('relu')(down6)

        # 8

        up7 = layers.UpSampling2D((2, 2))(down6)

        # 16

        up6 = layers.concatenate([up7, plus_6], axis=3)
        up6 = layers.Conv2D(64, (3, 3), padding='same')(up6)
        up6 = layers.Activation('relu')(up6)
        up6 = layers.Conv2D(32, (3, 3), padding='same')(up6)
        up6 = layers.concatenate([up6, up7], axis=3)
        up6 = layers.Activation('relu')(up6)
        up6 = layers.UpSampling2D((2, 2))(up6)

        # 32

        up5 = layers.concatenate([up6, plus_5], axis=3)
        up5 = layers.Conv2D(64, (3, 3), padding='same')(up5)
        up5 = layers.Activation('relu')(up5)
        up5 = layers.Conv2D(32, (3, 3), padding='same')(up5)
        up5 = layers.concatenate([up5, up6], axis=3)
        up5 = layers.Activation('relu')(up5)
        up5 = layers.UpSampling2D((2, 2))(up5)

        # 64

        up4 = layers.concatenate([up5, plus_4], axis=3)
        up4 = layers.Conv2D(64, (3, 3), padding='same')(up4)
        up4 = layers.Activation('relu')(up4)
        up4 = layers.Conv2D(32, (3, 3), padding='same')(up4)
        up4 = layers.concatenate([up4, up5], axis=3)
        up4 = layers.Activation('relu')(up4)
        up4 = layers.UpSampling2D((2, 2))(up4)

        # 128

        up3 = layers.concatenate([up4, plus_3], axis=3)
        up3 = layers.Conv2D(64, (3, 3), padding='same')(up3)
        up3 = layers.Activation('relu')(up3)
        up3 = layers.Conv2D(32, (3, 3), padding='same')(up3)
        up3 = layers.concatenate([up3, up4], axis=3)
        up3 = layers.Activation('relu')(up3)
        up3 = layers.UpSampling2D((2, 2))(up3)

        # 256

        up2 = layers.concatenate([up3, plus_2], axis=3)
        up2 = layers.Conv2D(64, (3, 3), padding='same')(up2)
        up2 = layers.Activation('relu')(up2)
        up2 = layers.Conv2D(32, (3, 3), padding='same')(up2)
        up2 = layers.concatenate([up2, up3], axis=3)
        up2 = layers.Activation('relu')(up2)
        up2 = layers.UpSampling2D((2, 2))(up2)

        # 512

        up1 = layers.concatenate([up2, plus_1], axis=3)
        up1 = layers.Conv2D(64, (3, 3), padding='same')(up1)
        up1 = layers.Activation('relu')(up1)
        up1 = layers.Conv2D(32, (3, 3), padding='same')(up1)
        up1 = layers.concatenate([up1, up2], axis=3)
        up1 = layers.Activation('relu')(up1)
        up1 = layers.UpSampling2D((2, 2))(up1)

        # 1024

        classify = layers.Conv2D(num_classes, (1, 1), activation='sigmoid')(up1)

        model = Model(inputs=inputs, outputs=classify)

        return model
