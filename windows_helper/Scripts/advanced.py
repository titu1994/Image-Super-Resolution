from keras.callbacks import Callback
from keras.regularizers import ActivityRegularizer
from keras import backend as K

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

class TVRegularizer(ActivityRegularizer):
    """ Enforces smoothness in image output. """

    def __init__(self, img_width, img_height, weight=2e-8):
        super(TVRegularizer, self).__init__()
        self.img_width = img_width
        self.img_height = img_height
        self.weight = weight
        self.uses_learning_phase = False

    def __call__(self, loss):
        x = self.layer.output
        assert K.ndim(x) == 4
        if K.image_dim_ordering() == 'th':
            a = K.square(x[:, :, :self.img_width - 1, :self.img_height - 1] - x[:, :, 1:, :self.img_height - 1])
            b = K.square(x[:, :, :self.img_width - 1, :self.img_height - 1] - x[:, :, :self.img_width - 1, 1:])
        else:
            a = K.square(x[:, :self.img_width - 1, :self.img_height - 1, :] - x[:, 1:, :self.img_height - 1, :])
            b = K.square(x[:, :self.img_width - 1, :self.img_height - 1, :] - x[:, :self.img_width - 1, 1:, :])
        loss += self.weight * K.mean(K.sum(K.pow(a + b, 1.25)))
        return loss

    def get_config(self):
        return {'name' : self.__class__.__name__,
                'img_width' : self.img_width,
                'img_height' : self.img_height,
                'weight' : self.weight}