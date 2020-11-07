import itertools

from keras.layers import Activation, Reshape, Lambda, concatenate, dot, add
from keras.layers import Conv1D, Conv2D, Conv3D
from keras.layers import MaxPool1D

from keras.callbacks import Callback, TensorBoard
from keras.engine.topology import Layer
from keras import backend as K

''' Callbacks '''
class HistoryCheckpoint(Callback):
    '''Callback that records events
        into a `History` object.

        It then saves the history after each epoch into a file.
        To read the file into a python dict:
            history = {}
            with open(filename, "r") as f:
                history = eval(f.read())

        This may be unsafe since eval() will evaluate any string
        A safer alternative:

        import ast

        history = {}
        with open(filename, "r") as f:
            history = ast.literal_eval(f.read())

    '''

    def __init__(self, filename):
        super(Callback, self).__init__()
        self.filename = filename

    def on_train_begin(self, logs={}):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs={}):
        self.epoch.append(epoch)
        for k, v in logs.items():
            if k not in self.history:
                self.history[k] = []
            self.history[k].append(v)

        with open(self.filename, "w") as f:
            f.write(str(self.history))


'''
Below is a modification to the TensorBoard callback to perform 
batchwise writing to the tensorboard, instead of only at the end
of the batch.
'''
class TensorBoardBatch(TensorBoard):
    def __init__(self, log_dir='./logs',
                 histogram_freq=0,
                 batch_size=32,
                 write_graph=True,
                 write_grads=False,
                 write_images=False,
                 embeddings_freq=0,
                 embeddings_layer_names=None,
                 embeddings_metadata=None):
        super(TensorBoardBatch, self).__init__(log_dir,
                                               histogram_freq=histogram_freq,
                                               batch_size=batch_size,
                                               write_graph=write_graph,
                                               write_grads=write_grads,
                                               write_images=write_images,
                                               embeddings_freq=embeddings_freq,
                                               embeddings_layer_names=embeddings_layer_names,
                                               embeddings_metadata=embeddings_metadata)

        # conditionally import tensorflow iff TensorBoardBatch is created
        self.tf = __import__('tensorflow')
        self.global_step = 1

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}

        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = self.tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, self.global_step)
        self.global_step += 1

        self.writer.flush()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = self.tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, self.global_step)

        self.global_step += 1
        self.writer.flush()


''' Theano Backend function '''

def depth_to_scale(x, scale, output_shape, dim_ordering=K.image_dim_ordering(), name=None):
    ''' Uses phase shift algorithm [1] to convert channels/depth for spacial resolution '''

    import theano.tensor as T

    scale = int(scale)

    if dim_ordering == "tf":
        x = x.transpose((0, 3, 1, 2))
        out_row, out_col, out_channels = output_shape
    else:
        out_channels, out_row, out_col = output_shape

    b, k, r, c = x.shape
    out_b, out_k, out_r, out_c = b, k // (scale * scale), r * scale, c * scale

    out = K.reshape(x, (out_b, out_k, out_r, out_c))

    for channel in range(out_channels):
        channel += 1

        for i in range(out_row):
            for j in range(out_col):
                a = i // scale #T.floor(i / scale).astype('int32')
                b = j // scale #T.floor(j / scale).astype('int32')
                d = channel * scale * (j % scale) + channel * (i % scale)

                T.set_subtensor(out[:, channel - 1, i, j], x[:, d, a, b], inplace=True)

    if dim_ordering == 'tf':
        out = out.transpose((0, 2, 3, 1))

    return out


''' Theano Backend function '''
def depth_to_scale_th(input, scale, channels):
    ''' Uses phase shift algorithm [1] to convert channels/depth for spacial resolution '''
    import theano.tensor as T

    b, k, row, col = input.shape
    output_shape = (b, channels, row * scale, col * scale)

    out = T.zeros(output_shape)
    r = scale

    for y, x in itertools.product(range(scale), repeat=2):
        out = T.inc_subtensor(out[:, :, y::r, x::r], input[:, r * y + x :: r * r, :, :])

    return out


''' Tensorflow Backend Function '''
def depth_to_scale_tf(input, scale, channels):
    try:
        import tensorflow as tf
    except ImportError:
        print("Could not import Tensorflow for depth_to_scale operation. Please install Tensorflow or switch to Theano backend")
        exit()

    def _phase_shift(I, r):
        ''' Function copied as is from https://github.com/Tetrachrome/subpixel/blob/master/subpixel.py'''

        bsize, a, b, c = I.get_shape().as_list()
        bsize = tf.shape(I)[0]  # Handling Dimension(None) type for undefined batch dim
        X = tf.reshape(I, (bsize, a, b, r, r))
        X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
        X = tf.split(1, a, X)  # a, [bsize, b, r, r]
        X = tf.concat(2, [tf.squeeze(x) for x in X])  # bsize, b, a*r, r
        X = tf.split(1, b, X)  # b, [bsize, a*r, r]
        X = tf.concat(2, [tf.squeeze(x) for x in X])  # bsize, a*r, b*r
        return tf.reshape(X, (bsize, a * r, b * r, 1))

    if channels > 1:
        Xc = tf.split(3, 3, input)
        X = tf.concat(3, [_phase_shift(x, scale) for x in Xc])
    else:
        X = _phase_shift(input, scale)
    return X

'''
Implementation is incomplete. Use lambda layer for now.
'''

class SubPixelUpscaling(Layer):

    def __init__(self, r, channels, **kwargs):
        super(SubPixelUpscaling, self).__init__(**kwargs)

        self.r = r
        self.channels = channels

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        if K.backend() == "theano":
            y = depth_to_scale_th(x, self.r, self.channels)
        else:
            y = depth_to_scale_tf(x, self.r, self.channels)
        return y

    def get_output_shape_for(self, input_shape):
        if K.image_dim_ordering() == "th":
            b, k, r, c = input_shape
            return (b, self.channels, r * self.r, c * self.r)
        else:
            b, r, c, k = input_shape
            return (b, r * self.r, c * self.r, self.channels)


''' Non Local Blocks '''

def non_local_block(ip, computation_compression=2, mode='embedded'):
    channel_dim = 1 if K.image_data_format() == 'channels_first' else -1
    ip_shape = K.int_shape(ip)

    if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
        raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')

    dim1, dim2, dim3 = None, None, None

    if len(ip_shape) == 3:  # time series data
        rank = 3
        batchsize, dim1, channels = ip_shape

    elif len(ip_shape) == 4:  # image data
        rank = 4

        if channel_dim == 1:
            batchsize, channels, dim1, dim2 = ip_shape
        else:
            batchsize, dim1, dim2, channels = ip_shape

    elif len(ip_shape) == 5:  # Video / Voxel data
        rank = 5

        if channel_dim == 1:
            batchsize, channels, dim1, dim2, dim3 = ip_shape
        else:
            batchsize, dim1, dim2, dim3, channels = ip_shape

    else:
        raise ValueError('Input dimension has to be either 3 (temporal), 4 (spatial) or 5 (spatio-temporal)')

    if mode == 'gaussian':  # Gaussian instantiation
        x1 = Reshape((-1, channels))(ip)  # xi
        x2 = Reshape((-1, channels))(ip)  # xj
        f = dot([x1, x2], axes=2)
        f = Activation('softmax')(f)

    elif mode == 'dot':  # Dot instantiation
        # theta path
        theta = _convND(ip, rank, channels // 2)
        theta = Reshape((-1, channels // 2))(theta)

        # phi path
        phi = _convND(ip, rank, channels // 2)
        phi = Reshape((-1, channels // 2))(phi)

        f = dot([theta, phi], axes=2)

        # scale the values to make it size invariant
        if batchsize is not None:
            f = Lambda(lambda z: 1./ batchsize * z)(f)
        else:
            f = Lambda(lambda z: 1. / 128 * z)(f)


    elif mode == 'concatenate':  # Concatenation instantiation
        raise NotImplemented('Concatenation mode has not been implemented yet')

    else:  # Embedded Gaussian instantiation
        # theta path
        theta = _convND(ip, rank, channels // 2)
        theta = Reshape((-1, channels // 2))(theta)

        # phi path
        phi = _convND(ip, rank, channels // 2)
        phi = Reshape((-1, channels // 2))(phi)

        if computation_compression > 1:
            # shielded computation
            phi = MaxPool1D(computation_compression)(phi)

        f = dot([theta, phi], axes=2)
        f = Activation('softmax')(f)

    # g path
    g = _convND(ip, rank, channels // 2)
    g = Reshape((-1, channels // 2))(g)

    if computation_compression > 1 and mode == 'embedded':
        # shielded computation
        g = MaxPool1D(computation_compression)(g)

    # compute output path
    y = dot([f, g], axes=[2, 1])

    # reshape to input tensor format
    if rank == 3:
        y = Reshape((dim1, channels // 2))(y)
    elif rank == 4:
        if channel_dim == -1:
            y = Reshape((dim1, dim2, channels // 2))(y)
        else:
            y = Reshape((channels // 2, dim1, dim2))(y)
    else:
        if channel_dim == -1:
            y = Reshape((dim1, dim2, dim3, channels // 2))(y)
        else:
            y = Reshape((channels // 2, dim1, dim2, dim3))(y)

    # project filters
    y = _convND(y, rank, channels)

    # residual connection
    residual = add([ip, y])

    return residual


def _convND(ip, rank, channels):
    assert rank in [3, 4, 5], "Rank of input must be 3, 4 or 5"

    if rank == 3:
        x = Conv1D(channels, 1, padding='same', use_bias=False)(ip)
    elif rank == 4:
        x = Conv2D(channels, (1, 1), padding='same', use_bias=False)(ip)
    else:
        x = Conv3D(channels, (1, 1, 1), padding='same', use_bias=False)(ip)
    return x
