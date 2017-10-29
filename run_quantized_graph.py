import tensorflow as tf
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm


PREVIEW_HEIGHT = 720
PREVIEW_WIDTH = 1280
INPUT_TENSOR_NAME = 'rgb_preview_input'
FINAL_TENSOR_NAME = 'rgb_output_blended'
QUANTIZED_PATH = 'tf_files/logged_quantized.pb'

img_fns = sorted(glob('data/data_road/training/image_2/*.png'))

with tf.Session() as sess:
    with tf.gfile.FastGFile(QUANTIZED_PATH, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

    input_tensor = sess.graph.get_tensor_by_name(
        INPUT_TENSOR_NAME + ':0')
    output_tensor = sess.graph.get_tensor_by_name(
        FINAL_TENSOR_NAME + ':0')

    for img_fn in tqdm(img_fns[::10]):
        img = cv2.imread(img_fn)
        img = cv2.resize(img, (PREVIEW_WIDTH, PREVIEW_HEIGHT))
        pimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pimg = np.expand_dims(pimg, axis=0)
        out = sess.run(output_tensor, feed_dict={input_tensor: pimg}).squeeze()
