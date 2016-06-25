from keras.models import Model
from keras.layers import Input, merge
from keras.layers.convolutional import Convolution2D
import keras.callbacks as callbacks
import numpy as np
import keras.backend as K
import img_utils

path_X = r"output_images_X\\"
path_Y = r"output_images_Y\\"

def PSNRLoss(y_true, y_pred):
    """
    PSNR is Peek Signal to Noise Ratio, which is similar to mean squared error.

    It can be calculated as
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE)

    When providing an unscaled input, MAXp = 255. Therefore 20 * log10(255)== 48.1308036087.
    However, since we are scaling our input, MAXp = 1. Therefore 20 * log10(1) = 0.
    Thus we remove that component completely and only compute the remaining MSE component.
    """
    return -10. * np.log10(K.mean(K.square(y_pred - y_true)))

class LearningModel(object):

    def __init__(self):
        self.model = None

        self.n1 = 64
        self.n2 = 32

    def create_model(self, height=33, width=33, channels=3, load_weights=False) -> Model:
        pass

    def fit(self, trainX, trainY, batch_size=128, nb_epochs=100) -> Model:
        if self.model == None: self.create_model()

    def evaluate(self):
        if self.model == None: self.create_model(load_weights=True)

        trainX, trainY = img_utils.loadImages()

        error = self.model.evaluate(trainX[-2000:], trainY[-2000:])
        print("Mean Squared Error (Compared to bilinear upscaled version) : ", error[0])
        print("Peak Signal to Noise Ratio (Compared to bilinear upscaled version) : ", error[1])

    def upscale(self, img_path, scale_factor=2, save_intermediate=False, suffix="scaled", verbose=True):
        import os
        import theano
        from scipy.misc import imread, imresize, imsave

        # Flag that may cause crash if algo_fwd = 'time_once'
        theano.config.dnn.conv.algo_fwd = 'small'

        # Read image
        img = imread(img_path, mode='RGB')
        height, width = img.shape[0], img.shape[1]
        if verbose: print("Old Size : ", img.shape)

        # Pre Upscale
        img = imresize(img, (height * scale_factor, width * scale_factor))
        height, width = img.shape[0], img.shape[1]
        if verbose: print("New Size : ", img.shape)

        # Transpose and Process image
        img_conv = img.transpose((2, 0, 1)).astype('float64') / 255
        img_conv = np.expand_dims(img_conv, axis=0)

        model = self.create_model(height, width, load_weights=True)
        print("Model loaded.")

        # Create prediction for image
        result = model.predict(img_conv, batch_size=1)
        print("Model finished upscaling.")

        # Evaluate agains bilinear upscaled image. Just to measure if a good result was obtained.
        print("Evaluating results.")
        error = model.evaluate(result, img_conv)
        print("Mean Squared Error (Compared to bilinear upscaled version) : ", error[0])
        print("Peak Signal to Noise Ratio (Compared to bilinear upscaled version) : ", error[1])

        # Deprocess
        result = result.reshape((3, height, width))
        result = result.transpose((1, 2, 0)).astype('float64') * 255
        result = np.clip(result, 0, 255).astype('uint8')

        path = os.path.splitext(img_path)
        filename = path[0] + "_" + suffix + "(%dx)" % (scale_factor) + path[1]

        print("Saving image.")
        # Save intermediate bilinear scaled image is needed for comparison.
        if save_intermediate:
            fn = path[0] + "_intermediate" + path[1]
            imsave(fn, img)

        imsave(filename, result)

class ImageSuperResolutionModel(LearningModel):

    def __init__(self):
        super(ImageSuperResolutionModel, self).__init__()

        self.f1 = 9
        self.f2 = 1
        self.f3 = 5

    def create_model(self, height=33, width=33, channels=3, load_weights=False):
        """
            Creates a model to be used to scale images of specific height and width.
        """
        init = Input(shape=(channels, height, width))

        x = Convolution2D(self.n1, self.f1, self.f1, activation='relu', border_mode='same', name='level1')(init)
        x = Convolution2D(self.n2, self.f2, self.f2, activation='relu', border_mode='same', name='level2')(x)

        out = Convolution2D(channels, self.f3, self.f3, border_mode='same', name='output')(x)

        model = Model(init, out)
        model.compile(optimizer='adadelta', loss='mse', metrics=[PSNRLoss])
        if load_weights: model.load_weights("SR Weights.h5")

        self.model = model
        return model

    def fit(self, trainX, trainY, batch_size=128, nb_epochs=100) -> Model:
        super(ImageSuperResolutionModel, self).fit(trainX, trainY, batch_size, nb_epochs)

        self.model.fit(trainX, trainY, batch_size=batch_size, nb_epoch=nb_epochs,
              callbacks=[callbacks.ModelCheckpoint("SR Weights.h5", monitor='val_PSNRLoss', save_best_only=True, mode='max')],
              validation_split=2000./38400)

        return self.model

class ExpantionSuperResolution(LearningModel):

    def __init__(self):
        super(ExpantionSuperResolution, self).__init__()

        self.f1 = 9
        self.f2_1 = 1
        self.f2_2 = 3
        self.f2_3 = 5
        self.f3 = 5

    def create_model(self, height=33, width=33, channels=3, load_weights=False):
        """
            Creates a model to be used to scale images of specific height and width.
        """
        init = Input(shape=(channels, height, width))

        x = Convolution2D(self.n1, self.f1, self.f1, activation='relu', border_mode='same', name='level1')(init)

        x1 = Convolution2D(self.n2, self.f2_1, self.f2_1, activation='relu', border_mode='same', name='lavel1_1')(x)
        x2 = Convolution2D(self.n2, self.f2_2, self.f2_2, activation='relu', border_mode='same', name='lavel1_2')(x)
        x3 = Convolution2D(self.n2, self.f2_3, self.f2_3, activation='relu', border_mode='same', name='lavel1_3')(x)

        x = merge([x1, x2, x3], mode='ave')

        out = Convolution2D(channels, self.f3, self.f3, border_mode='same', name='output')(x)

        model = Model(init, out)
        model.compile(optimizer='adadelta', loss='mse', metrics=[PSNRLoss])
        if load_weights: model.load_weights("Expantion SR Weights.h5")

        self.model = model
        return model

    def fit(self, trainX, trainY, batch_size=128, nb_epochs=100) -> Model:
        super(ExpantionSuperResolution, self).fit(trainX, trainY, batch_size, nb_epochs)

        self.model.fit(trainX, trainY, batch_size=batch_size, nb_epoch=nb_epochs,
            callbacks=[callbacks.ModelCheckpoint("Expantion SR Weights.h5", monitor='val_PSNRLoss', save_best_only=True, mode='max')],
            validation_split=2000. / 38400)

        return self.model