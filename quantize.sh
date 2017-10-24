#!/bin/bash
TF_ROOT=$HOME/tensorflow
INPUT_TENSOR_NAME="rgb_preview_input"
FINAL_TENSOR_NAME="rgb_output_blended"
FROZEN_GRAPH="tf_files/frozen.pb"
QUANTIZED_GRAPH="tf_files/range_quantized_graph.pb"

# Just quantize
$TF_ROOT/bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
--in_graph=$FROZEN_GRAPH \
--out_graph="tf_files/quantized_graph.pb" \
--inputs=$INPUT_TENSOR_NAME \
--outputs=$FINAL_TENSOR_NAME \
--transforms='add_default_attributes
strip_unused_nodes(type=float, shape="1,720,1280,3")
remove_nodes(op=Identity, op=CheckNumerics)
fuse_pad_and_conv
fuse_resize_and_conv
fuse_resize_pad_and_conv
flatten_atrous_conv
fold_constants
fold_batch_norms
fold_old_batch_norms
quantize_weights
quantize_nodes
merge_duplicate_nodes
strip_unused_nodes
sort_by_execution_order
'
# obfuscate_names

# create graph that logs the requantized ranges
$TF_ROOT/bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
--in_graph="tf_files/quantized_graph.pb" \
--out_graph="tf_files/logged_quantized.pb" \
--inputs=$INPUT_TENSOR_NAME \
--outputs=$FINAL_TENSOR_NAME \
--transforms='insert_logging(op=RequantizationRange, show_name=true, message="__requant_min_max:")'

# run the graph and log the ranges
python3 run_quantized_graph.py 2> logged_ranges.txt

# finally freeze the requantized ranges
$TF_ROOT/bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
--in_graph="tf_files/quantized_graph.pb" \
--out_graph=$QUANTIZED_GRAPH \
--inputs=$INPUT_TENSOR_NAME \
--outputs=$FINAL_TENSOR_NAME \
--transforms='freeze_requantization_ranges(min_max_log_file=logged_ranges.txt)'
