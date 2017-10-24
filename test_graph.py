import tensorflow as tf
import numpy as np
import cv2
from time import time
from os.path import basename as bn

INPUT_TENSOR_NAME = 'image_input'
FINAL_TENSOR_NAME = 'lambda_4/ResizeBilinear'
FREEZED_PATH = 'tf_files/frozen.pb'
OPTIMIZED_PATH = 'tf_files/optimized.pb'
IMAGE_SHAPE = (160, 576)
put_text = lambda img, text: cv2.putText(img, text, (10, 50), cv2.FONT_HERSHEY_DUPLEX, 2.0, (255, 127, 127), 3, cv2.LINE_AA)  # noqa


def time_run(graph_path, device='/gpu:0', num_runs=3):
    tf.reset_default_graph()
    with tf.device(device):
        with tf.Session(
                config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False)) as sess:
            with tf.gfile.FastGFile(OPTIMIZED_PATH, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')

            input_tensor = sess.graph.get_tensor_by_name(
                INPUT_TENSOR_NAME + ':0')
            output_tensor = sess.graph.get_tensor_by_name(
                FINAL_TENSOR_NAME + ':0')

            img = cv2.imread('data/data_road/training/image_2/umm_000045.png')
            img = cv2.resize(img, (IMAGE_SHAPE[1], IMAGE_SHAPE[0]))
            pimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pimg = np.float32(pimg) / 127.5 - 1.0
            pimg = np.expand_dims(pimg, axis=0)
            # warm up
            out = sess.run(
                output_tensor, feed_dict={input_tensor: pimg}).squeeze()
            tic = time()
            for t in range(num_runs):
                out = sess.run(output_tensor,
                               feed_dict={input_tensor: pimg}).squeeze()
            toc = time()
            duration = (toc - tic) / num_runs
            print("One forward pass for `%s` on `%s` took: %.4f ms"
                  % (bn(graph_path), device, duration * 1000))
            pred = np.uint8(out.argmax(axis=-1))
            pred_rgb = np.dstack((0 * pred, 255 * pred, 0 * pred))
            res_img = cv2.addWeighted(img, 0.6, pred_rgb, 0.4, 0.0)
            img_fn = (bn(graph_path) + device + '.png').replace('/', '_')
            put_text(res_img, img_fn)
            cv2.imwrite(img_fn, res_img)


time_run(FREEZED_PATH, device='/gpu:0')
time_run(OPTIMIZED_PATH, device='/gpu:0')
time_run(FREEZED_PATH, device='/cpu:0')
time_run(OPTIMIZED_PATH, device='/cpu:0')
