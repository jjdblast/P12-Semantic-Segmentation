import tensorflow as tf
sess = tf.Session()
from tensorflow import gfile
from tensorflow.python.framework import graph_util
import os
import shutil
from keras import backend as K
K.set_session(sess)
K.set_learning_phase(0)
from tensorflow.tools.graph_transforms import TransformGraph
import argparse
from seg_mobilenet import SegMobileNet
from IPython import embed

ALPHA = 0.6
ROAD_COLOR = [0, 255, 0]
INPUT_HEIGHT = 160
INPUT_WIDTH = 576
CROP_HEIGHT = 360
PREVIEW_HEIGHT = 720
PREVIEW_WIDTH = 1280
INPUT_TENSOR_NAME = 'rgb_preview_input'
FINAL_TENSOR_NAME = 'rgb_output_blended'
FREEZED_PATH = 'tf_files/frozen.pb'
OPTIMIZED_PATH = 'tf_files/optimized.pb'

parser = argparse.ArgumentParser()
parser.add_argument(
    "--weight_path",
    type=str,
    default='experiment_005/ep-098-val_loss-0.0847.hdf5',
    help="Path to hdf5 weight file.")
args = parser.parse_args()

if not os.path.exists('tf_files'):
    os.makedirs('tf_files')
else:
    shutil.rmtree('tf_files')
    os.makedirs('tf_files')

rgb_preview_input = tf.placeholder(
    tf.float32,
    shape=[None, PREVIEW_HEIGHT, PREVIEW_WIDTH, 3],
    name=INPUT_TENSOR_NAME)
rgb_preview_cropped = rgb_preview_input[:, CROP_HEIGHT:, :, :]
preview_resized = tf.image.resize_bilinear(
    rgb_preview_cropped, [INPUT_HEIGHT, INPUT_WIDTH], align_corners=True)
scaled_input = (preview_resized / 127.5 - 1.0)
model = SegMobileNet(160, 576, num_classes=2)
model.load_weights(args.weight_path)
output = model(scaled_input)
output_resized = tf.image.resize_bilinear(
    output, [PREVIEW_HEIGHT - CROP_HEIGHT, PREVIEW_WIDTH], align_corners=True)
output_pred = tf.cast(tf.argmax(output_resized, axis=-1), tf.float32)
output_pred = tf.pad(output_pred, ((0, 0), (CROP_HEIGHT, 0), (0, 0)))
# output_pred = (1.0 - output_pred)
output_pred *= ALPHA
output_pred = tf.stack((output_pred, output_pred, output_pred), axis=-1)
blended_pred = tf.add(
    (1.0 - output_pred) * rgb_preview_input,
    output_pred * ROAD_COLOR,
    name=FINAL_TENSOR_NAME)

names = [n.name for n in sess.graph.as_graph_def().node]

tf.summary.FileWriter('ok', graph=sess.graph)
# embed()
optimized_graph_def = graph_util.convert_variables_to_constants(
    sess, sess.graph.as_graph_def(), [FINAL_TENSOR_NAME])

with gfile.FastGFile(FREEZED_PATH, 'wb') as f:
    f.write(optimized_graph_def.SerializeToString())

print("Starting graph optimization ... ")
transforms = [
    'strip_unused_nodes(type=float, shape="1,160,576,3")',
    'remove_nodes(op=Identity, op=CheckNumerics)',
    'fold_constants(ignore_errors=false)',
    'fold_batch_norms',
    'fuse_resize_pad_and_conv',
    'fuse_resize_and_conv',
    'fuse_pad_and_conv',
    'fold_old_batch_norms',
    'remove_device',
    'round_weights(num_steps=256)',
    'strip_unused_nodes']

for transform in transforms:
    try:
        print("Starting transform: `%s` ... " % transform)
        optimized_graph_def = TransformGraph(
            optimized_graph_def,
            [INPUT_TENSOR_NAME],
            [FINAL_TENSOR_NAME],
            [transform])
    except:
        print('Transform failed: `%s`' % transform)

tf.summary.FileWriter('opt_log', graph_def=optimized_graph_def)
print("Wrote optimized graph to `%s` ... " % 'opt_log')

with gfile.FastGFile(OPTIMIZED_PATH, 'wb') as f:
    f.write(optimized_graph_def.SerializeToString())

print("Done! Wrote results to `%s`." % 'tf_files')
