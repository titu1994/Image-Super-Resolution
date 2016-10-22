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

def depth_to_scale(x, scale, channels=3, dim_ordering=K.image_dim_ordering(), name=None):
    ''' Uses phase shift algorithm [1] to convert channels/depth for spacial resolution '''

    import theano.tensor as T

    scale = int(scale)

    if dim_ordering == "tf":
        x = x.transpose((0, 3, 1, 2))

    b, k, r, c = x.shape
    out_b, out_k, out_r, out_c = b, k // (scale * scale), r * scale, c * scale

    out = K.reshape(x, (out_b, out_k, out_r, out_c))

    for channel in range(channels):
        for i in range(scale):
            for j in range(scale):
                channel += 1
                a = T.floor(i / scale).astype('int32')
                b = T.floor(j / scale).astype('int32')
                d = channel * scale * (j % scale) + channel * (i % scale)

                T.set_subtensor(out[:, channel - 1, i, j], x[:, d, a, b], inplace=True)

    if dim_ordering == 'tf':
        out = out.transpose((0, 2, 3, 1))

    return out


class SubpixelConvolution2D(Layer):

    def __init__(self, r):
        super(SubpixelConvolution2D, self).__init__()
        self.r = r

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        x = depth_to_scale(x, self.r)
        return x

    def get_output_shape_for(self, input_shape):
        if K.image_dim_ordering() == "th":
            b, k, r, c = input_shape
            return (b, k / (self.r * self.r), r * self.r, c * self.r)
        else:
            b, r, c, k = input_shape
            return (b, r * self.r, c * self.r, k / (self.r * self.r))