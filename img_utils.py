from __future__ import print_function, division, absolute_import

import numpy as np
from scipy.misc import imsave, imread, imresize
from sklearn.feature_extraction.image import reconstruct_from_patches_2d, extract_patches_2d
from scipy.ndimage.filters import gaussian_filter

from keras import backend as K

import os
import time

'''
_image_scale_multiplier is a special variable which is used to alter image size.

The default image size is 32x32. If a true upscaling model is used, then the input image size is 16x16,
which not offer adequate training samples.
'''
_image_scale_multiplier = 1

img_size = 256 * _image_scale_multiplier
stride = 16 * _image_scale_multiplier

assert (img_size ** 2) % (stride ** 2) == 0, "Number of images generated from strided subsample of the image needs to be \n" \
                                             "a positive integer. Change stride such that : \n" \
                                             "(img_size ** 2) / (stride ** 2) is a positive integer."

input_path = r"input_images/"
validation_path = r"val_images/"

validation_set5_path = validation_path + "set5/"
validation_set14_path = validation_path + "set14/"

base_dataset_dir = os.path.expanduser("~") + "/Image Super Resolution Dataset/"

output_path = base_dataset_dir + "train_images/train/"
validation_output_path = base_dataset_dir + r"train_images/validation/"

if not os.path.exists(output_path):
    os.makedirs(output_path)

def transform_images(directory, output_directory, scaling_factor=2, max_nb_images=-1, true_upscale=False):
    index = 1

    if not os.path.exists(output_directory + "X/"):
        os.makedirs(output_directory + "X/")

    if not os.path.exists(output_directory + "y/"):
        os.makedirs(output_directory + "y/")

    # For each image in input_images directory
    nb_images = len([name for name in os.listdir(directory)])

    if max_nb_images != -1:
        print("Transforming %d images." % max_nb_images)
    else:
        assert max_nb_images <= nb_images, "Max number of images must be less than number of images in path"
        print("Transforming %d images." % (nb_images))

    if nb_images == 0:
        print("Extract the training images or images from imageset_91.zip (found in the releases of the project) "
              "into a directory with the name 'input_images'")
        print("Extract the validation images or images from set5_validation.zip (found in the releases of the project) "
              "into a directory with the name 'val_images'")
        exit()

    for file in os.listdir(directory):
        img = imread(directory + file, mode='RGB')

        # Resize to 256 x 256
        img = imresize(img, (img_size, img_size))

        # Create patches
        hr_patch_size = (16 * scaling_factor * _image_scale_multiplier)
        nb_hr_images = (img_size ** 2) // (stride ** 2)

        hr_samples = np.empty((nb_hr_images, hr_patch_size, hr_patch_size, 3))

        image_subsample_iterator = subimage_generator(img, stride, hr_patch_size, nb_hr_images)

        stride_range = np.sqrt(nb_hr_images).astype(int)

        i = 0
        for j in range(stride_range):
            for k in range(stride_range):
                hr_samples[i, :, :, :] = next(image_subsample_iterator)
                i += 1

        lr_patch_size = 16 * _image_scale_multiplier

        t1 = time.time()
        # Create nb_hr_images 'X' and 'Y' sub-images of size hr_patch_size for each patch
        for i in range(nb_hr_images):
            ip = hr_samples[i]
            # Save ground truth image X
            imsave(output_directory + "/y/" + "%d_%d.png" % (index, i + 1), ip)

            # Apply Gaussian Blur to Y
            op = gaussian_filter(ip, sigma=0.5)

            # Subsample by scaling factor to Y
            op = imresize(op, (lr_patch_size, lr_patch_size), interp='bicubic')

            if not true_upscale:
                # Upscale by scaling factor to Y
                op = imresize(op, (hr_patch_size, hr_patch_size), interp='bicubic')

            # Save Y
            imsave(output_directory + "/X/" + "%d_%d.png" % (index, i+1), op)

        print("Finished image %d in time %0.2f seconds. (%s)" % (index, time.time() - t1, file))
        index += 1

        if max_nb_images > 0 and index >= max_nb_images:
            print("Transformed maximum number of images. ")
            break

    print("Images transformed. Saved at directory : %s" % (output_directory))


def image_count():
    return len([name for name in os.listdir(output_path + "X/")])


def val_image_count():
    return len([name for name in os.listdir(validation_output_path + "X/")])


def subimage_generator(img, stride, patch_size, nb_hr_images):
    for _ in range(nb_hr_images):
        for x in range(0, img_size - patch_size, stride):
            for y in range(0, img_size - patch_size, stride):
                subimage = img[x : x + patch_size, y : y + patch_size, :]

                yield subimage


def make_patches(x, scale, patch_size, upscale=True, verbose=1):
    '''x shape: (num_channels, rows, cols)'''
    height, width = x.shape[:2]
    if upscale: x = imresize(x, (height * scale, width * scale))
    patches = extract_patches_2d(x, (patch_size, patch_size))
    return patches


def combine_patches(in_patches, out_shape, scale):
    '''Reconstruct an image from these `patches`'''
    recon = reconstruct_from_patches_2d(in_patches, out_shape)
    return recon

def image_generator(directory, scale_factor=2, target_shape=None, channels=3, small_train_images=False, shuffle=True,
                    batch_size=32, seed=None):
    if not target_shape:
        if small_train_images:
            if K.image_dim_ordering() == "th":
                image_shape = (channels, 16 * _image_scale_multiplier, 16 * _image_scale_multiplier)
                y_image_shape = (channels, 16 * scale_factor * _image_scale_multiplier,
                                 16 * scale_factor * _image_scale_multiplier)
            else:
                image_shape = (16 * _image_scale_multiplier, 16 * _image_scale_multiplier, channels)
                y_image_shape = (16 * scale_factor * _image_scale_multiplier,
                                 16 * scale_factor * _image_scale_multiplier, channels)
        else:
            if K.image_dim_ordering() == "th":
                image_shape = (channels, 16 * scale_factor * _image_scale_multiplier, 16 * scale_factor * _image_scale_multiplier)
                y_image_shape = image_shape
            else:
                image_shape = (16 * scale_factor * _image_scale_multiplier, 16 * scale_factor * _image_scale_multiplier,
                               channels)
                y_image_shape = image_shape
    else:
        if small_train_images:
            if K.image_dim_ordering() == "th":
                y_image_shape = (3,) + target_shape

                target_shape = (target_shape[0] * _image_scale_multiplier // scale_factor,
                                target_shape[1] * _image_scale_multiplier // scale_factor)
                image_shape = (3,) + target_shape
            else:
                y_image_shape = target_shape + (channels,)

                target_shape = (target_shape[0] * _image_scale_multiplier // scale_factor,
                                target_shape[1] * _image_scale_multiplier // scale_factor)
                image_shape = target_shape + (channels,)
        else:
            if K.image_dim_ordering() == "th":
                image_shape = (channels,) + target_shape
                y_image_shape = image_shape
            else:
                image_shape = target_shape + (channels,)
                y_image_shape = image_shape

    file_names = [f for f in sorted(os.listdir(directory + "X/"))]
    X_filenames = [os.path.join(directory, "X", f) for f in file_names]
    y_filenames = [os.path.join(directory, "y", f) for f in file_names]

    nb_images = len(file_names)
    print("Found %d images." % nb_images)

    index_generator = _index_generator(nb_images, batch_size, shuffle, seed)

    while 1:
        index_array, current_index, current_batch_size = next(index_generator)

        batch_x = np.zeros((current_batch_size,) + image_shape)
        batch_y = np.zeros((current_batch_size,) + y_image_shape)

        for i, j in enumerate(index_array):
            x_fn = X_filenames[j]
            img = imread(x_fn, mode='RGB')
            if small_train_images:
                img = imresize(img, (16 * _image_scale_multiplier, 16 * _image_scale_multiplier))
            img = img.astype('float32') / 255.

            if K.image_dim_ordering() == "th":
                batch_x[i] = img.transpose((2, 0, 1))
            else:
                batch_x[i] = img

            y_fn = y_filenames[j]
            img = imread(y_fn, mode="RGB")
            img = img.astype('float32') / 255.

            if K.image_dim_ordering() == "th":
                batch_y[i] = img.transpose((2, 0, 1))
            else:
                batch_y[i] = img

        yield (batch_x, batch_y)

def _index_generator(N, batch_size=32, shuffle=True, seed=None):
    batch_index = 0
    total_batches_seen = 0

    while 1:
        if seed is not None:
            np.random.seed(seed + total_batches_seen)

        if batch_index == 0:
            index_array = np.arange(N)
            if shuffle:
                index_array = np.random.permutation(N)

        current_index = (batch_index * batch_size) % N

        if N >= current_index + batch_size:
            current_batch_size = batch_size
            batch_index += 1
        else:
            current_batch_size = N - current_index
            batch_index = 0
        total_batches_seen += 1

        yield (index_array[current_index: current_index + current_batch_size],
               current_index, current_batch_size)


def smooth_gan_labels(y):
    assert len(y.shape) == 2, "Needs to be a binary class"
    y = np.asarray(y, dtype='int')
    Y = np.zeros(y.shape, dtype='float32')

    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            if y[i, j] == 0:
                Y[i, j] = np.random.uniform(0.0, 0.3)
            else:
                Y[i, j] = np.random.uniform(0.7, 1.2)

    return Y


if __name__ == "__main__":
    # Transform the images once, then run the main code to scale images

    # Change scaling factor to increase the scaling factor
    scaling_factor = 2

    # Set true_upscale to True to generate smaller training images that will then be true upscaled.
    # Leave as false to create same size input and output images
    true_upscale = False

    transform_images(input_path, output_path, scaling_factor=scaling_factor, max_nb_images=-1,
                     true_upscale=true_upscale)
    transform_images(validation_set5_path, validation_output_path, scaling_factor=scaling_factor, max_nb_images=-1,
                     true_upscale=true_upscale)
    pass