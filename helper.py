import re
import random
import numpy as np
import os.path
import scipy.misc
import shutil
import zipfile
import time
import tensorflow as tf
from glob import glob
from urllib.request import urlretrieve
from keras.utils import get_file
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from IPython import embed


class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def maybe_download_pretrained_vgg(data_dir):
    """
    Download and extract pretrained vgg model if it doesn't exist
    :param data_dir: Directory to download the model to
    """
    vgg_filename = 'vgg.zip'
    vgg_path = os.path.join(data_dir, 'vgg')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]

    missing_vgg_files = [
        vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        # Clean vgg dir
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)

        # Download vgg
        print('Downloading pre-trained vgg model...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',  # noqa
                os.path.join(
                    vgg_path,
                    vgg_filename),
                pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(vgg_path, vgg_filename))


def maybe_download_mobilenet_weights(alpha_text='1_0', rows=224):
    base_weight_path = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.6/'  # noqa
    model_name = 'mobilenet_%s_%d_tf_no_top.h5' % (alpha_text, rows)
    weigh_path = base_weight_path + model_name
    weight_path = get_file(model_name,
                           weigh_path,
                           cache_subdir='models')
    return weight_path


def gen_batches_functions(data_folder, image_shape,
                          train_augmentation_fn=None,
                          val_augmentation_fn=None):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    image_paths = sorted(
        glob(os.path.join(data_folder, 'image_2', '*.png')))[:]
    train_paths, val_paths = train_test_split(
        image_paths, test_size=0.1, random_state=21)

    def get_batches_fn(batch_size, image_paths, augmentation_fn=None):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        label_fns = glob(os.path.join(
            data_folder, 'gt_image_2', '*_road_*.png'))
        label_paths = {
            re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
            for path in label_fns}

        background_color = np.array([255, 0, 0])
        random.shuffle(image_paths)
        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            gt_images = []
            for image_file in image_paths[batch_i:batch_i + batch_size]:
                gt_image_file = label_paths[os.path.basename(image_file)]

                image = scipy.misc.imresize(
                    scipy.misc.imread(image_file, mode='RGB'), image_shape)

                gt_image = scipy.misc.imresize(
                    scipy.misc.imread(gt_image_file), image_shape)

                gt_bg = np.all(gt_image == background_color, axis=2)
                if augmentation_fn:
                    image, gt_bg = augmentation_fn(image, gt_bg)

                gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
                gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)
                images.append(image)
                gt_images.append(gt_image)

            yield np.array(images) / 127.5 - 1.0, np.array(gt_images)

    train_batches_fn = lambda batch_size: get_batches_fn(batch_size, train_paths, augmentation_fn=train_augmentation_fn)  # noqa
    val_batches_fn = lambda batch_size: get_batches_fn(batch_size, val_paths, augmentation_fn=val_augmentation_fn)  # noqa  

    return train_batches_fn, val_batches_fn


def gen_test_output(
        sess,
        logits,
        image_pl,
        data_folder,
        learning_phase,
        image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    for image_file in sorted(
            glob(os.path.join(data_folder, 'image_2', '*.png')))[:]:
        image = scipy.misc.imresize(
            scipy.misc.imread(image_file, mode='RGB'), image_shape)
        pimg = image / 127.5 - 1.0
        im_softmax = sess.run(
            tf.nn.softmax(logits),
            {image_pl: [pimg],
             learning_phase: 0})
        im_softmax = im_softmax[:, 1].reshape(
            image_shape[0], image_shape[1])
        segmentation = (
            im_softmax > 0.5).reshape(
            image_shape[0],
            image_shape[1],
            1)
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)

        yield os.path.basename(image_file), np.array(street_im)


def save_inference_samples(
        runs_dir,
        data_dir,
        sess,
        image_shape,
        logits,
        learning_phase,
        input_image):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(
        sess, logits, input_image, os.path.join(
            data_dir, 'data_road/testing'), learning_phase, image_shape)
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)
