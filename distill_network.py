from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras import backend as K

import models
import img_utils
from advanced import HistoryCheckpoint, TensorBoardBatch

scale_factor = 2
batchsize = 128
nb_epochs = 50

teacher_model = models.DistilledResNetSR(scale_factor)
teacher_model.create_model(load_weights=True)
teacher_model.model.summary()

print("\n\n\n")

teacher_output_tensor = teacher_model.model.layers[-1].output

for layer in teacher_model.model.layers:
    layer.trainable = False

student_model = models.DistilledResNetSR(scale_factor)
student_model.create_model()
student_model.model.summary()

def zero_loss(y_true, y_pred):
    return 0 * y_true

def gram_matrix(x):
    assert K.ndim(x) == 4

    with K.name_scope('gram_matrix'):
        if K.image_data_format() == "channels_first":
            batch, channels, width, height = K.int_shape(x)
            features = K.batch_flatten(x)
        else:
            batch, width, height, channels = K.int_shape(x)
            features = K.batch_flatten(K.permute_dimensions(x, (0, 3, 1, 2)))

        gram = K.dot(features, K.transpose(features)) / (channels * width * height)
    return gram

joint_model = Model(inputs=[student_model.model.input, teacher_model.model.input],
                    outputs=student_model.model.output)

student_output_tensor = joint_model.layers[-1].output

# teacher - student l2 loss
with K.name_scope('l2_loss'):
    l2_weight = 1e-3
    teacher_student_loss = K.sum(K.square(teacher_output_tensor - student_output_tensor))  # l2 norm of difference
joint_model.add_loss(l2_weight * teacher_student_loss)

# perceptual loss
with K.name_scope('perceptual_loss'):
    perceptual_weight = 2.
    perceptual_loss = K.sum(K.square(gram_matrix(teacher_output_tensor) - gram_matrix(student_output_tensor)))
joint_model.add_loss(perceptual_weight * perceptual_loss)

joint_model.compile(optimizer='adam', loss=zero_loss)

# train student model using teacher model
samples_per_epoch = img_utils.image_count()
val_count = img_utils.val_image_count()

weight_path = 'weights/joint_model (%s) %dX.h5' % (teacher_model.model_name, scale_factor)
history_fn = 'Joint_model_training.txt'

train_path = img_utils.output_path
validation_path = img_utils.validation_output_path
path_X = img_utils.output_path + "X/"
path_Y = img_utils.output_path + "y/"

callback_list = [ModelCheckpoint(weight_path, monitor='val_loss', save_best_only=True,
                                 mode='min', save_weights_only=True, verbose=2),
                 TensorBoardBatch('./distillation_logs_%s/' % teacher_model.model_name),
                 HistoryCheckpoint(history_fn),
                 ]

print("Training model : %s" % ("Joint Model"))
joint_model.fit_generator(img_utils.image_generator(train_path, scale_factor=scale_factor,
                                                    small_train_images=teacher_model.type_true_upscaling,
                                                    batch_size=batchsize,
                                                    nb_inputs=2),  # 2 input joint model
                         steps_per_epoch=samples_per_epoch // batchsize + 1,
                         epochs=nb_epochs, callbacks=callback_list,
                         validation_data=img_utils.image_generator(validation_path,
                                                                   scale_factor=scale_factor,
                                                                   small_train_images=teacher_model.type_true_upscaling,
                                                                   batch_size=val_count,
                                                                   nb_inputs=2),  # 2 input joint model
                         validation_steps=1)

student_model.model.save_weights('weights/student_model_final %dX.h5' % scale_factor, overwrite=True)