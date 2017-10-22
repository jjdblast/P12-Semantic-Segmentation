# Semantic Segmentation
### Overview
This project is part of the Udacity Car-ND. Originally, it uses a VGG-frontend. However, VGG is old, slow and uses too much memory. E.g. a single (160, 576) image already requires 4GB GPU memory. I therefore switched to the [MobileNet](https://arxiv.org/abs/1704.04861) architecture. All stride-16 depthwise convolutions were replaced with dilated depthwise convolutions and the two final stride-32 layers were removed. This is similar to what was done in [Multi-Scale Context Aggregation by Dilated Convolutions](https://arxiv.org/abs/1511.07122).
Though, I added skip-connections for stride-8 and stride-4 (adding stride-2 gave no better results). 
The model uses the [Keras MobileNet implementation](https://github.com/fchollet/keras/blob/master/keras/applications/mobilenet.py) and training is done with
[TensorFlow](https://www.tensorflow.org/).
Since the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) provides no offical training/validation splits I used 10% of the training data for validation. Images are downscaled to half resolution.

[//]: # (Image References)
[image1]: ./res/loss_curves.png
[image2]: ./res/augmentation_methods_overview.png
[image3]: ./res/latest_run.png
[image4]: ./res/benchmark_results.png
[image5]: ./res/highway.gif
[image6]: ./res/padded_icon.png

#### Data augmentation
The dataset is rather small i.e. only 289 training images. Thus, I used a few augmentation methods to generate more data. These methods include: rotation, flipping, blurring and changing the illumination of the scene (see `augmentation.py`).
An example is given in the following image:
![alt text][image2]


#### Quantitative Results
Training for 100 epochs results in the following loss curves:
![alt text][image1]
I ran a few experiments with different learning rate schedules, varying amounts of data augmentation and changed dilation rates but the results did not change that much. Moreover, there was no big difference in training from scratch vs. using ImageNet weights.
By default logs are saved to the `log` directory. To visualize them start tensorboard via `tensorboard --logdir log`.

#### Qualitative Results
Here are a few predictions @~92% validation mIoU:
![alt text][image3]
It can be seen that shadows are handled quite well. Road vs. sidewalk still leaves room for improvments (e.g. bottom left). More results can be found in the `latest_run` directory.

### Inference Optimization
There are plenty of [methods for inference optimization](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/graph_transforms/README.md) already implemented in tensorflow. I am using the `fold_constants`, `fold_batch_norms` and `round_weights` transforms (see `export_for_mobile.py`).
These methods are explained in more detail on the [Pete Warden blog](https://petewarden.com/2017/06/22/what-ive-learned-about-neural-network-quantization/).
After all, the (zipped) optimized graph is only 1.8M.
To optimize your graph run:
```
python3 export_for_mobile.py --weight_path checkpoint/ep-045-val_loss-0.0123.hdf5
```
You can check and benchmark the optimized graph with:
```
python3 test_graph.py
```
You should see that you get nice (tiny) speed-ups on the CPU (GPU) while the results stay the same:
```
One forward pass for `freezed.pb` on `/gpu:0` took: 20.6018 ms
One forward pass for `optimized.pb` on `/gpu:0` took: 20.5921 ms
One forward pass for `freezed.pb` on `/cpu:0` took: 163.9354 ms
One forward pass for `optimized.pb` on `/cpu:0` took: 146.2478 ms
```
Though, I only checked them visually:
![alt text][image4]

### Generalization
KITTI only includes urban images so I tested it on a short highway scene captured with my smartphone. The results are not super accurate. Probably due to different camera parameters or maybe just not enough data.
![alt text][image5]

### Get the app
![alt text][image6]

You can download the latest version of Roady from [Google Play](https://play.google.com/store/apps/details?id=org.steffen.roady).
**Note**: Most of the app code comes from these two codelabs:
- [TensorFlow for Poets 2](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets-2/index.html?index=..%2F..%2Findex#0)
- [Android & TensorFlow: Artistic Style Transfer](https://codelabs.developers.google.com/codelabs/tensorflow-style-transfer-android/index.html?index=..%2F..%2Findex#0)


### Setup
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [Keras](https://keras.io/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
 - [OpenCV](https://opencv.org/)

##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training and test images.

### Start
Run the following command to start the project:
```
python3 main.py
```
