import itertools

from keras.callbacks import Callback
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