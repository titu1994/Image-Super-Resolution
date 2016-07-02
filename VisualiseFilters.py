import numpy as np
import time
from scipy.misc import imsave
from models import PSNRLoss
import models


from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras.layers.convolutional import Convolution2D

import seaborn as sns
sns.set_style("white")

"""
Code from the Keras examples which details how to visualize filters.
"""

img_size = 128

layer_name = "conv1"

def deprocessImage(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

f1, f2, f3 = 9, 1, 5
n1, n2 = 64, 32

c = 3 # Number of channels in input image

init = Input(shape=(c, img_size, img_size))

model = models.ImageSuperResolutionModel().create_model(load_weights=True)

firstLayer = model.layers[0]
input_img = firstLayer.input

# get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict = dict([(layer.name, layer) for layer in model.layers])

kept_filters = []
for filter_index in range(0, n1):
    print('Processing filter %d' % filter_index)
    start_time = time.time()

    # we build a loss function that maximizes the activation
    # of the nth filter of the layer considered
    layer_output = layer_dict[layer_name].output
    loss = K.mean(layer_output[:, filter_index, :, :])

    # we compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, input_img)[0]

    # normalization trick: we normalize the gradient
    grads = normalize(grads)

    # this function returns the loss and grads given the input picture
    iterate = K.function([input_img], [loss, grads])

    # step size for gradient ascent
    step = 1.

    # we start from a gray image with some random noise
    input_img_data = np.random.random((1, 3, img_size, img_size)) * 20 + 128.

    # we run gradient ascent for 20 steps
    for i in range(20):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

        print('Current loss value:', loss_value)
        if loss_value <= 0.:
            # some filters get stuck to 0, we can skip them
            break

    # decode the resulting input image
    if loss_value > 0:
        img = deprocessImage(input_img_data[0])
        kept_filters.append((img, loss_value))
    end_time = time.time()
    print('Filter %d processed in %ds' % (filter_index, end_time - start_time))

# we will stich the best 64 filters on a 8 x 8 grid.
n = 7

# the filters that have the highest loss are assumed to be better-looking.
# we will only keep the top 64 filters.
kept_filters.sort(key=lambda x: x[1], reverse=True)
kept_filters = kept_filters[:n * n]

# build a black picture with enough space for
# our 8 x 8 filters of size 128 x 128, with a 5px margin in between
margin = 5
width = n * img_size + (n - 1) * margin
height = n * img_size + (n - 1) * margin
stitched_filters = np.zeros((width, height, 3))

# fill the picture with our saved filters
for i in range(n):
    for j in range(n):
        img, loss = kept_filters[i * n + j]
        stitched_filters[(img_size + margin) * i: (img_size + margin) * i + img_size,
                         (img_size + margin) * j: (img_size + margin) * j + img_size, :] = img

# save the result to disk
imsave('ISR %s stitched_filters_%dx%dx%dx%d.png' % (layer_name, n, n, img_size, img_size), stitched_filters)
