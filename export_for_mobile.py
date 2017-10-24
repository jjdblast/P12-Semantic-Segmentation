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

INPUT_TENSOR_NAME = 'image_input'
FINAL_TENSOR_NAME = 'lambda_4/ResizeBilinear'
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

model = SegMobileNet(160, 576, num_classes=2)
model.load_weights(args.weight_path)

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
    'remove_device',
    'fold_old_batch_norms',
    'fold_constants(ignore_errors=false)',
    'round_weights(num_steps=256)',
]

# 'fuse_convolutions',
for transform in transforms:
    print("Starting transform: `%s` ... " % transform)
    optimized_graph_def = TransformGraph(
        optimized_graph_def,
        [INPUT_TENSOR_NAME],
        [FINAL_TENSOR_NAME],
        [transform])

tf.summary.FileWriter('opt_log', graph_def=optimized_graph_def)
print("Wrote optimized graph to `%s` ... " % 'opt_log')

with gfile.FastGFile(OPTIMIZED_PATH, 'wb') as f:
    f.write(optimized_graph_def.SerializeToString())

print("Done! Wrote results to `%s`." % 'tf_files')
