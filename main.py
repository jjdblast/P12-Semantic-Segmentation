import os.path
import os
import shutil
import tensorflow as tf
from keras import backend as K
import helper
import warnings
from distutils.version import LooseVersion
from seg_mobilenet import SegMobileNet
import project_tests as tests
from tqdm import tqdm
import numpy as np
from IPython import embed
from augmentation import rotate_both, flip_both, blur_both, illumination_change_both  # noqa
# https://keras.io/backend/
KERAS_TRAIN = 1
KERAS_TEST = 0

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), \
    'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(
        tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn(
        'No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, loss)
    """
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label = tf.reshape(correct_label, (-1, num_classes))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=correct_label, logits=logits))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    # optimizer = tf.train.MomentumOptimizer(
    #     learning_rate=learning_rate, momentum=0.9)
    train_op = optimizer.minimize(loss)

    predicted_label = tf.argmax(logits, axis=-1)
    sparse_correct_label = tf.argmax(correct_label, axis=-1)
    with tf.variable_scope("iou") as scope:
        iou, iou_op = tf.metrics.mean_iou(
            sparse_correct_label, predicted_label, num_classes)
    metric_vars = [v for v in tf.local_variables()
                   if v.name.split('/')[0] == 'iou']
    metric_reset_ops = tf.variables_initializer(metric_vars)
    return logits, train_op, loss, iou, iou_op, metric_reset_ops


# tests.test_optimize(optimize)


def train_nn(
        sess,
        epochs,
        batch_size,
        train_batches_fn, val_batches_fn,
        train_op,
        cross_entropy_loss,
        input_image, correct_label,
        learning_rate, learning_rate_val, decay, learning_phase,
        iou_op, iou,
        metric_reset_ops, update_ops,
        model):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.
           Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    train_loss_summary = tf.placeholder(tf.float32)
    val_loss_summary = tf.placeholder(tf.float32)
    train_iou_summary = tf.placeholder(tf.float32)
    val_iou_summary = tf.placeholder(tf.float32)

    tf.summary.scalar("train_loss", train_loss_summary)
    tf.summary.scalar("train_iou", train_iou_summary)
    tf.summary.scalar("val_loss", val_loss_summary)
    tf.summary.scalar("val_iou", val_iou_summary)
    tf.summary.scalar("learning_rate", learning_rate)

    summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter('log', graph=sess.graph)

    all_vars = tf.global_variables()
    optimizer_variables = [v for v in all_vars
                           if v not in model.updates and
                           v not in model.trainable_weights]
    # embed()
    sess.run(metric_reset_ops)
    sess.run(tf.variables_initializer(optimizer_variables))
    for var in all_vars:
        try:
            sess.run(var)
        except:
            print("Err ", var)
            embed()
    epoch_pbar = tqdm(range(epochs))
    for epoch in epoch_pbar:
        # train
        train_loss = 0.0
        iteration_counter = 0
        for image, label in train_batches_fn(batch_size):
            fetches = [cross_entropy_loss, train_op, iou_op] + update_ops
            feed_dict = {
                input_image: image, correct_label: label,
                learning_rate: learning_rate_val, learning_phase: KERAS_TRAIN,
            }
            loss_val, *_ = sess.run(  # noqa
                fetches,
                feed_dict=feed_dict)
            # embed()
            train_loss += loss_val
            iteration_counter += 1

        learning_rate_val = learning_rate_val / (1.0 + decay * epoch)
        train_iou = sess.run(iou)
        train_loss /= iteration_counter
        val_loss = 0.0
        iteration_counter = 0
        sess.run(metric_reset_ops)
        # val
        for image, label in val_batches_fn(batch_size):
            feed_dict = {
                input_image: image, correct_label: label,
                learning_phase: KERAS_TEST,
            }
            *_, loss_val = sess.run(
                [iou_op, cross_entropy_loss], feed_dict=feed_dict)
            val_loss += loss_val
            iteration_counter += 1

        val_iou = sess.run(iou)
        val_loss /= iteration_counter
        epoch_pbar.write(
            "Epoch %03d: loss: %.4f mIoU: %.4f val_loss: %.4f val_mIoU: %.4f"
            % (epoch, train_loss, train_iou, val_loss, val_iou))
        summary_val = sess.run(
            summary, feed_dict={train_loss_summary: train_loss,
                                val_loss_summary: val_loss,
                                train_iou_summary: train_iou,
                                val_iou_summary: val_iou,
                                learning_rate: learning_rate_val})
        writer.add_summary(summary_val, epoch)
        if epoch % 2 == 0:
            weight_path = 'checkpoint/ep-%03d-val_loss-%.4f.hdf5' \
                          % (epoch, val_loss)
            model.save_weights(weight_path)


def augmentation_fn(image, label):
    """Wrapper for augmentation methods
    """
    image = np.uint8(image)
    label = np.uint8(label)
    image, label = flip_both(image, label, p=0.5)
    image, label = rotate_both(image, label, p=0.5, ignore_label=1)
    image, label = blur_both(image, label, p=0.5)
    image, label = illumination_change_both(image, label, p=0.5)
    return image, label == 1


# tests.test_train_nn(train_nn)
def run():
    from_scratch = False
    num_classes = 2
    image_shape = (160, 576)
    learning_rate_val = 0.001
    epochs = 100
    decay = learning_rate_val / (2 * epochs)
    batch_size = 18
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)
    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/
    if not from_scratch:
        weight_path = helper.maybe_download_mobilenet_weights()
    if os.path.exists('checkpoint'):
        shutil.rmtree('checkpoint')
    os.makedirs('checkpoint')

    with K.get_session() as sess:
        # Create function to get batches
        train_batches_fn, val_batches_fn = helper.gen_batches_functions(
            os.path.join(data_dir, 'data_road/training'), image_shape,
            train_augmentation_fn=augmentation_fn)

        learning_phase = K.learning_phase()
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        correct_label = tf.placeholder(
            tf.float32,
            shape=[None, image_shape[0], image_shape[1], num_classes],
            name='correct_label')

        model = SegMobileNet(
            image_shape[0], image_shape[1], num_classes=num_classes)
        # this initializes the keras variables
        sess = K.get_session()
        if not from_scratch:
            model.load_weights(weight_path, by_name=True)
        input_image = model.input
        logits = model.output
        # https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html  # noqa
        update_ops = model.updates

        logits, train_op, loss, iou, iou_op, metric_reset_ops = optimize(
            logits, correct_label, learning_rate, num_classes)

        train_nn(sess, epochs, batch_size,
                 train_batches_fn, val_batches_fn,
                 train_op, loss, input_image,
                 correct_label,
                 learning_rate, learning_rate_val, decay, learning_phase,
                 iou_op, iou, metric_reset_ops,
                 update_ops,
                 model)

        helper.save_inference_samples(
            runs_dir, data_dir, sess, image_shape,
            logits, learning_phase, input_image)
        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
