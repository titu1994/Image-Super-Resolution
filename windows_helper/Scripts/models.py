from keras.models import Model
from keras.layers import Input, merge, BatchNormalization, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D
from keras import backend as K
import keras.callbacks as callbacks
import keras.optimizers as optimizers

from advanced import HistoryCheckpoint, TVRegularizer
import img_utils

import numpy as np
import os
import time

train_path = r"train_images/train/"
validation_path = img_utils.validation_output_path
path_X = img_utils.output_path + "/X/"
path_Y = img_utils.output_path + "/y/"

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

def psnr(y_true, y_pred):
    assert y_true.shape == y_pred.shape, "Cannot calculate PSNR. Input shapes not same." \
                                         " y_true shape = %s, y_pred shape = %s" % (str(y_true.shape),
                                                                                   str(y_pred.shape))

    return -10. * np.log10(np.mean(np.square(y_pred - y_true)))


class BaseSuperResolution(object):

    def __init__(self, model_name):
        """
        Base model to provide a standard interface of adding Super Resolution models
        """
        self.model = None # type: Model
        self.model_name = model_name

        self.n1 = 64
        self.n2 = 32

        self.denoise_models = []

        self.evaluation_func = None

    def create_model(self, height=32, width=32, channels=3, load_weights=False, batch_size=128,
                     small_train_images=False) -> Model:
        """
        Subclass dependent implementation.
        """
        pass

    def fit(self, scale_factor, weight_fn, batch_size=128, nb_epochs=100, small_train_images=False,
                                save_history=True, history_fn="Model History.txt") -> Model:
        """
        Standard method to train any of the models.
        """
        samples_per_epoch = img_utils.image_count()
        val_count = img_utils.val_image_count()
        if self.model == None: self.create_model(batch_size=batch_size)

        callback_list = [callbacks.ModelCheckpoint(weight_fn, monitor='val_PSNRLoss', save_best_only=True,  mode='max', save_weights_only=True),]
        if save_history: callback_list.append(HistoryCheckpoint(history_fn))

        print("Training model : %s" % (self.__class__.__name__))
        self.model.fit_generator(img_utils.image_generator(train_path, scale_factor=scale_factor,
                                                           small_train_images=small_train_images,
                                                           batch_size=batch_size),
                                 samples_per_epoch=samples_per_epoch,
                                 nb_epoch=nb_epochs, callbacks=callback_list,
                                 validation_data=img_utils.image_generator(validation_path,
                                                                          scale_factor=scale_factor,
                                                                          small_train_images=small_train_images,
                                                                          batch_size=batch_size),
                                 nb_val_samples=val_count)

        return self.model

    def evaluate(self, validation_dir, scale_factor, target_size=256, small_train_images=False):
        """
        Evaluates the model on the Set5 Validation images
        """
        if self.model == None: self.create_model(load_weights=True, small_train_images=small_train_images)

        if self.evaluation_func is None:
            self.evaluation_func = K.function([self.model.layers[0].input],
                                              [self.model.layers[-1].output])

        predict_path = "val_predict/"
        if not os.path.exists(predict_path):
            os.makedirs(predict_path)

        image_fns = [name for name in os.listdir(validation_dir)]
        nb_images = len(image_fns)
        print("Validating %d images" % (nb_images))

        total_psnr = 0.0

        for impath in os.listdir(validation_dir):
            t1 = time.time()

            # Input image
            y = img_utils.imread(validation_dir + impath, mode='RGB')
            width, height, _ = y.shape

            if self.model_name in self.denoise_models:
                # Denoise models require precise width and height, divisible by 4

                if ((width // scale_factor) % 4 != 0) or ((height // scale_factor) % 4 != 0):
                    width = ((width // scale_factor) // 4) * 4 * scale_factor
                    height = ((height // scale_factor) // 4) * 4 * scale_factor

                    print("Model %s require the image size to be divisible by 4. New image size = (%d, %d)" % \
                                                                                (self.model_name, width, height))

                    y = img_utils.imresize(y, (width, height), interp='bicubic')

            y = y.astype('float32') / 255.
            y = np.expand_dims(y, axis=0)

            x_width = width if not small_train_images else width // scale_factor
            x_height = height if not small_train_images else height // scale_factor

            x_temp = y.copy()
            img = img_utils.gaussian_filter(x_temp[0], sigma=0.01)
            img = img_utils.imresize(img, (x_width // scale_factor, x_height // scale_factor), interp='bicubic')

            if not small_train_images:
                img = img_utils.imresize(img, (x_width, x_height), interp='bicubic')

            x = np.expand_dims(img, axis=0)

            if K.image_dim_ordering() == "th":
                x = x.transpose((0, 3, 1, 2))
                y = y.transpose((0, 3, 1, 2))

            y_pred = self.evaluation_func([x])[0][0]

            psnr_val = psnr(y[0], np.clip(y_pred, 0, 255) / 255)
            total_psnr += psnr_val

            t2 = time.time()
            print("Validated image : %s, Time required : %0.2f, PSNR value : %0.4f" % (impath, t2 - t1, psnr_val))

            generated_path = predict_path + "%s_generated.png" % (os.path.splitext(impath)[0])

            if K.image_dim_ordering() == "th":
                y_pred = y_pred.transpose((1, 2, 0))

            y_pred = np.clip(y_pred, 0, 255).astype('uint8')
            img_utils.imsave(generated_path, y_pred)

        print("Average PRNS value of validation images = %00.4f" % (total_psnr / nb_images))

    def upscale(self, img_path, scale_factor=2, save_intermediate=False, return_image=False, suffix="scaled",
                patch_size=8, mode="patch", verbose=True, evaluate=True):
        """
        Standard method to upscale an image.

        :param img_path:  path to the image
        :param scale_factor: scale factor can be any value, usually is an integer
        :param save_intermediate: saves the intermediate upscaled image (bilinear upscale)
        :param return_image: returns a image of shape (height, width, channels).
        :param suffix: suffix of upscaled image
        :param patch_size: size of each patch grid
        :param verbose: whether to print messages
        :param evaluate: evaluate the upscaled image on the original image.
        """
        import os
        from scipy.misc import imread, imresize, imsave

        # Destination path
        path = os.path.splitext(img_path)
        filename = path[0] + "_" + suffix + "(%dx)" % (scale_factor) + path[1]

        # Read image
        scale_factor = int(scale_factor)
        true_img = imread(img_path, mode='RGB')
        init_width, init_height = true_img.shape[0], true_img.shape[1]
        if verbose: print("Old Size : ", true_img.shape)
        if verbose: print("New Size : (%d, %d, 3)" % (init_height * scale_factor, init_width * scale_factor))

        img_height, img_width = 0, 0

        if mode == 'patch':
            # Create patches
            if self.model_name in self.denoise_models:
                if patch_size % 4 != 0:
                    print("Deep Denoise requires patch size which is multiple of 4.\nSetting patch_size = 8.")
                    patch_size = 8

            images = img_utils.make_patches(true_img, scale_factor, patch_size, verbose)

            nb_images = images.shape[0]
            img_width, img_height = images.shape[1], images.shape[2]
            print("Number of patches = %d, Patch Shape = (%d, %d)" % (nb_images, img_height, img_width))
        else:
            # Use full image for super resolution
            img_width, img_height = self.__match_denoise_size(denoise_models, img_height, img_width, init_height,
                                                              init_width, scale_factor)

            images = imresize(true_img, (img_width, img_height))
            images = np.expand_dims(images, axis=0)
            print("Image is reshaped to : (%d, %d, %d)" % (images.shape[1], images.shape[2], images.shape[3]))

        # Save intermediate bilinear scaled image is needed for comparison.
        intermediate_img = None
        if save_intermediate:
            if verbose: print("Saving intermediate image.")
            fn = path[0] + "_intermediate_" + path[1]
            intermediate_img = imresize(true_img, (init_width * scale_factor, init_height * scale_factor))
            imsave(fn, intermediate_img)

        # Transpose and Process images
        if K.image_dim_ordering() == "th":
            img_conv = images.transpose((0, 3, 1, 2)).astype(np.float32) / 255.
        else:
            img_conv = images.astype(np.float32) / 255.

        model = self.create_model(img_height, img_width, load_weights=True)
        if verbose: print("Model loaded.")

        # Create prediction for image patches
        result = model.predict(img_conv, batch_size=128, verbose=verbose)

        if verbose: print("De-processing images.")

         # Deprocess patches
        if K.image_dim_ordering() == "th":
            result = result.transpose((0, 2, 3, 1)).astype(np.float32) * 255.
        else:
            result = result.astype(np.float32) * 255.

        # Output shape is (original_width * scale, original_height * scale, nb_channels)
        if mode == 'patch':
            out_shape = (init_width * scale_factor, init_height * scale_factor, 3)
            result = img_utils.combine_patches(result, out_shape, scale_factor)
        else:
            result = result[0, :, :, :] # Access the 3 Dimensional image vector

        result = np.clip(result, 0, 255).astype('uint8')

        if verbose: print("\nCompleted De-processing image.")

        if return_image:
            # Return the image without saving. Useful for testing images.
            return result

        if verbose: print("Saving image.")
        imsave(filename, result)

        if evaluate:
            if verbose: print("Evaluating results.")
            # Convert initial image into correct format
            if intermediate_img is None:
                intermediate_img = imresize(true_img, (init_width * scale_factor, init_height * scale_factor))

            if mode == 'patch':
                intermediate_img = img_utils.make_patches(intermediate_img, scale_factor, patch_size, upscale=False)
            else:
                img_width, img_height = self.__match_denoise_size(denoise_models, img_height, img_width, init_height,
                                                                  init_width, scale_factor)

                intermediate_img = imresize(true_img, (img_width, img_height))
                intermediate_img = np.expand_dims(intermediate_img, axis=0)

            if K.image_dim_ordering() == "th":
                intermediate_img = intermediate_img.transpose((0, 3, 1, 2)).astype(np.float32) / 255.
            else:
                intermediate_img = intermediate_img.astype(np.float32) / 255.

            eval_model = self.create_model(img_height, img_width, load_weights=True)

            # Evaluating the initial image patches, which gets transformed to the output image, to the input image
            error = eval_model.evaluate(img_conv, intermediate_img, batch_size=128)
            print("\nMean Squared Error of %s  : " % (self.model_name), error[0])
            print("Peak Signal to Noise Ratio of %s : " % (self.model_name), error[1])

    def __match_denoise_size(self, denoise_models, img_height, img_width, init_height, init_width, scale_factor):
        if self.model_name in denoise_models:
            if ((init_height * scale_factor) % 4 != 0) or ((init_width * scale_factor) % 4 != 0):
                print("Deep Denoise requires image size which is multiple of 4.")

                img_height = ((init_height * scale_factor) // 4) * 4
                img_width = ((init_width * scale_factor) // 4) * 4
            else:
                img_height, img_width = init_height, init_width
        return img_height, img_width


class ImageSuperResolutionModel(BaseSuperResolution):

    def __init__(self):
        super(ImageSuperResolutionModel, self).__init__("Image SR")

        self.f1 = 9
        self.f2 = 1
        self.f3 = 5

    def create_model(self, height=32, width=32, channels=3, load_weights=False, batch_size=128,
                     small_train_images=False):
        """
            Creates a model to be used to scale images of specific height and width.
        """

        if K.image_dim_ordering() == "th":
            shape = (channels, width, height)
        else:
            shape = (width, height, channels)

        init = Input(shape=shape)

        x = Convolution2D(self.n1, self.f1, self.f1, activation='relu', border_mode='same', name='level1')(init)
        x = Convolution2D(self.n2, self.f2, self.f2, activation='relu', border_mode='same', name='level2')(x)

        out = Convolution2D(channels, self.f3, self.f3, border_mode='same', name='output')(x)

        model = Model(init, out)

        adam = optimizers.Adam(lr=1e-4)
        model.compile(optimizer=adam, loss='mse', metrics=[PSNRLoss])
        if load_weights: model.load_weights("weights/SR Weights.h5")

        self.model = model
        return model

    def fit(self, scale_factor, weight_fn="weights/SR Weights.h5", batch_size=128, nb_epochs=100, small_train_images=False,
                                save_history=True, history_fn="SRCNN History.txt"):
        return super(ImageSuperResolutionModel, self).fit(scale_factor, weight_fn, batch_size, nb_epochs,
                                                          small_train_images, save_history, history_fn)


class ExpantionSuperResolution(BaseSuperResolution):

    def __init__(self):
        super(ExpantionSuperResolution, self).__init__("Expanded Image SR")

        self.f1 = 9
        self.f2_1 = 1
        self.f2_2 = 3
        self.f2_3 = 5
        self.f3 = 5

    def create_model(self, height=32, width=32, channels=3, load_weights=False, batch_size=128,
                     small_train_images=False):
        """
            Creates a model to be used to scale images of specific height and width.
        """
        if K.image_dim_ordering() == "th":
            shape = (channels, width, height)
        else:
            shape = (width, height, channels)

        init = Input(shape=shape)

        x = Convolution2D(self.n1, self.f1, self.f1, activation='relu', border_mode='same', name='level1')(init)

        x1 = Convolution2D(self.n2, self.f2_1, self.f2_1, activation='relu', border_mode='same', name='lavel1_1')(x)
        x2 = Convolution2D(self.n2, self.f2_2, self.f2_2, activation='relu', border_mode='same', name='lavel1_2')(x)
        x3 = Convolution2D(self.n2, self.f2_3, self.f2_3, activation='relu', border_mode='same', name='lavel1_3')(x)

        x = merge([x1, x2, x3], mode='ave')

        out = Convolution2D(channels, self.f3, self.f3, activation='relu', border_mode='same', name='output')(x)

        model = Model(init, out)
        model.compile(optimizer='adadelta', loss='mse', metrics=[PSNRLoss])
        if load_weights: model.load_weights("weights/Expantion SR Weights.h5")

        self.model = model
        return model

    def fit(self, scale_factor, weight_fn="weights/Expantion SR Weights.h5", batch_size=128, nb_epochs=100, small_train_images=False,
                                save_history=True, history_fn="ESRCNN History.txt"):
        return super(ExpantionSuperResolution, self).fit(scale_factor, weight_fn, batch_size, nb_epochs,
                                                         small_train_images, save_history, history_fn)


class DenoisingAutoEncoderSR(BaseSuperResolution):

    def __init__(self):
        super(DenoisingAutoEncoderSR, self).__init__("Denoise AutoEncoder SR")

    def create_model(self, height=32, width=32, channels=3, load_weights=False, batch_size=128,
                     small_train_images=False):
        """
            Creates a model to remove / reduce noise from upscaled images.
        """
        from keras.layers.convolutional import Deconvolution2D

        assert height % 4 == 0, "Height of the image must be divisible by 4"
        assert width % 4 == 0, "Width of the image must be divisible by 4"

        if K.image_dim_ordering() == "th":
            shape = (channels, width, height)
        else:
            shape = (width, height, channels)

        init = Input(shape=shape)

        level1_1 = Convolution2D(self.n1, 3, 3, activation='relu', border_mode='same')(init)
        level2_1 = Convolution2D(self.n1, 3, 3, activation='relu', border_mode='same')(level1_1)

        level2_2 = Deconvolution2D(self.n1, 3, 3, activation='relu', output_shape=(None, channels, height, width), border_mode='same')(level2_1)
        level2 = merge([level2_1, level2_2], mode='sum')

        level1_2 = Deconvolution2D(self.n1, 3, 3, activation='relu', output_shape=(None, channels, height, width), border_mode='same')(level2)
        level1 = merge([level1_1, level1_2], mode='sum')

        decoded = Convolution2D(channels, 5, 5, activation='linear', border_mode='same')(level1)

        model = Model(init, decoded)
        model.compile(optimizer='adam', loss='mse', metrics=[PSNRLoss])
        if load_weights: model.load_weights("weights/Denoising AutoEncoder.h5")

        self.model = model
        return model

    def fit(self, scale_factor, weight_fn="weights/Denoising AutoEncoder.h5", batch_size=128, nb_epochs=100, small_train_images=False,
                                save_history=True, history_fn="DSRCNN History.txt"):
        return super(DenoisingAutoEncoderSR, self).fit(scale_factor, weight_fn, batch_size, nb_epochs,
                                                       small_train_images, save_history, history_fn)

class DeepDenoiseSR(BaseSuperResolution):

    def __init__(self):
        super(DeepDenoiseSR, self).__init__("Deep Denoise SR")

        # Treat this model as a denoising auto encoder
        # Force the fit, evaluate and upscale methods to take special care about image shape
        self.denoise_models.append(self.model_name)

        self.n1 = 64
        self.n2 = 128
        self.n3 = 256

    def create_model(self, height=32, width=32, channels=3, load_weights=False, batch_size=128, small_train_images=False):
        assert height % 4 == 0, "Height of the image must be divisible by 4"
        assert width % 4 == 0, "Width of the image must be divisible by 4"

        if K.image_dim_ordering() == "th":
            shape = (channels, width, height)
        else:
            shape = (width, height, channels)

        init = Input(shape=shape)

        c1 = Convolution2D(self.n1, 3, 3, activation='relu', border_mode='same')(init)
        c1 = Convolution2D(self.n1, 3, 3, activation='relu', border_mode='same')(c1)

        x = MaxPooling2D((2, 2))(c1)

        c2 = Convolution2D(self.n2, 3, 3, activation='relu', border_mode='same')(x)
        c2 = Convolution2D(self.n2, 3, 3, activation='relu', border_mode='same')(c2)

        x = MaxPooling2D((2, 2))(c2)

        c3 = Convolution2D(self.n3, 3, 3, activation='relu', border_mode='same')(x)

        x = UpSampling2D()(c3)

        c2_2 = Convolution2D(self.n2, 3, 3, activation='relu', border_mode='same')(x)
        c2_2 = Convolution2D(self.n2, 3, 3, activation='relu', border_mode='same')(c2_2)

        m1 = merge([c2, c2_2], mode='sum')
        m1 = UpSampling2D()(m1)

        c1_2 = Convolution2D(self.n1, 3, 3, activation='relu', border_mode='same')(m1)
        c1_2 = Convolution2D(self.n1, 3, 3, activation='relu', border_mode='same')(c1_2)

        m2 = merge([c1, c1_2], mode='sum')

        decoded = Convolution2D(channels, 5, 5, activation='linear', border_mode='same')(m2)

        model = Model(init, decoded)
        model.compile(optimizer='adam', loss='mse', metrics=[PSNRLoss])
        if load_weights: model.load_weights("weights/Deep Denoise Weights.h5")

        self.model = model
        return model

    def fit(self, scale_factor, weight_fn="weights/Deep Denoise Weights.h5", batch_size=128, nb_epochs=100, small_train_images=False,
                                save_history=True, history_fn="Deep DSRCNN History.txt"):
        super(DeepDenoiseSR, self).fit(scale_factor, weight_fn, batch_size, nb_epochs,
                                       small_train_images, save_history, history_fn)

class ResNetSR(BaseSuperResolution):

    def __init__(self):
        super(ResNetSR, self).__init__("ResNetSR")

        # Treat this model as a denoising auto encoder
        # Force the fit, evaluate and upscale methods to take special care about image shape
        self.denoise_models.append(self.model_name)

        self.n = 64

        self.mode = 2

    def create_model(self, height=32, width=32, channels=3, load_weights=False, batch_size=128, small_train_images=False):
        assert height % 4 == 0, "Height of the image must be divisible by 4"
        assert width % 4 == 0, "Width of the image must be divisible by 4"

        if K.image_dim_ordering() == "th":
            shape = (channels, width, height)
        else:
            shape = (width, height, channels)

        init = Input(shape=shape)

        x0 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='sr_res_conv1')(init)

        x1 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', subsample=(2, 2), name='sr_res_conv2')(x0)

        x2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', subsample=(2, 2), name='sr_res_conv3')(x1)

        x = self._residual_block(x2, 1)

        nb_residual = 14
        for i in range(nb_residual):
            x = self._residual_block(x, i + 2)

        x = UpSampling2D()(x)
        x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='sr_res_deconv1',)(x)

        x = merge([x, x1], mode='sum')

        x = UpSampling2D()(x)
        x = Convolution2D(64, 3, 3, activation='relu', border_mode='same',  name='sr_res_deconv2')(x)

        x = merge([x, x0], mode='sum')

        tv_regularizer = TVRegularizer(img_width=width, img_height=height, weight=2e-8)
        x = Convolution2D(3, 3, 3, activation="linear", border_mode='same', activity_regularizer=tv_regularizer,
                          name='sr_res_conv_final')(x)

        model = Model(init, x)

        adam = optimizers.Adam(lr=1e-4)
        model.compile(optimizer=adam, loss='mse', metrics=[PSNRLoss])
        if load_weights: model.load_weights("weights/ResNetSR.h5")

        self.model = model
        return model

    def _residual_block(self, ip, id):
        init = ip

        x = Convolution2D(64, 3, 3, activation='linear', border_mode='same',
                          name='sr_res_conv_' + str(id) + '_1')(ip)
        x = BatchNormalization(axis=1, mode=self.mode, name="sr_res_batchnorm_" + str(id) + "_1")(x)
        x = Activation('relu', name="sr_res_activation_" + str(id) + "_1")(x)

        x = Convolution2D(64, 3, 3, activation='linear', border_mode='same',
                          name='sr_res_conv_' + str(id) + '_2')(x)
        x = BatchNormalization(axis=1, mode=self.mode, name="sr_res_batchnorm_" + str(id) + "_2")(x)

        m = merge([x, init], mode='sum', name="sr_res_merge_" + str(id))

        return m

    def fit(self, scale_factor, weight_fn="weights/ResNetSR.h5", batch_size=128, nb_epochs=100, small_train_images=False,
                                save_history=True, history_fn="ResNetSR History.txt"):
        super(ResNetSR, self).fit(scale_factor, weight_fn, batch_size, nb_epochs,
                                  small_train_images, save_history, history_fn)
