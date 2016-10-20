from keras.callbacks import Callback
from keras.regularizers import ActivityRegularizer
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

def depth_to_scale(x, scale, dim_ordering=K.image_dim_ordering(), name=None):
    ''' Uses phase shift algorithm [1] to convert channels/depth for spacial resolution '''

    import theano.tensor as T

    if dim_ordering == "tf":
        x = x.transpose((0, 3, 1, 2))

    b, k, r, c = x.shape

    out = K.zeros((b, k / (scale * scale), r * scale, c * scale))

    for i in range(scale):
        for j in range(scale):
            T.set_subtensor(out[:, :, i :: scale, j :: scale],
                            x[:, scale * i + j :: scale * scale, :, :], inplace=True)

    if dim_ordering == 'tf':
        out = out.transpose((0, 2, 3, 1))

    return out