# Yoga_pose_recognizer
yoga pose recognizer is an algorithm that detects 5 yoga poses , based on a pre-trained model called movenet 

 1. [About Movenet](#about-movenet)
 2. [Implementation in tensorflow](#Implementation-in-tensorflow)
 3. [Example on a real Dataset](#Example-on-a-real-Dataset)

## About movenet 
MoveNet is a convolutional neural network that runs on RGB images and predicts human joint locations of a single person ,the model is designed to be run in the browser using tensorflow.js or on devices using TF-LITE in real-time , targeting movements/fitness  activities , two variants are presented 
 
    ● MoveNet.SinglePose.Lightning: A lower capacity model that can run >50FPS on most
    modern laptops while achieving good performance.
    ● MoveNet.SinglePose.Thunder: A higher capacity model that performs better prediction
    quality while still achieving real-time (>30FPS) speed
### Model Architecture
MobileNetV2 image feature extractor with Feature Pyramid Network(here is the paper https://arxiv.org/pdf/1612.03144.pdf) decoder (to stride of 4)
followed by CenterNet prediction heads with custom post-processing logic. Lightning uses depth multiplier 1.0 while Thunder uses depth multiplier 1.75 
![image](https://user-images.githubusercontent.com/47725118/130789918-65d968b8-5a5e-46a2-9129-8d9225435eb6.png)
MobileNetV2 architecture : 
### Inputs
A frame of video or an image, represented as an int32 tensor of shape: 192x192x3(Lightning) / 256x256x3(Thunder). Channels order: RGB with values in [0, 255].
### Outputs
A float32 tensor of shape [1, 1, 17, 3].<br>
● The first two channels of the last dimension represents the yx coordinates (normalized to image frame, i.e. range in [0.0, 1.0]) of the 17 keypoints (in the order of: [nose, left eye, right eye, left ear, right ear, left shoulder, right shoulder, left elbow, right elbow, left wrist,right wrist, left hip, right hip, left knee, right knee, left ankle, right ankle]).
● The third channel of the last dimension represents the prediction confidence scores of
each keypoint, also in the range [0.0, 1.0]..<br>
### Out-of-scope Use Cases 
● This model is not intended for detecting poses of multiple people in the image(still in progress by tensorflow team) , there are some accurate multipe-pose models like OpenPose ( here is the paper https://arxiv.org/pdf/1812.08008.pdf )  

## Implementation in tensorflow
you can find the implementations [here](https://tfhub.dev/s?tf-version=tf2&q=movenet)
```python
  # Import TF and TF Hub libraries.
  import tensorflow as tf
  import tensorflow_hub as hub

  # Load the input image.
  image_path = 'PATH_TO_YOUR_IMAGE'
  image = tf.io.read_file(image_path)
  image = tf.compat.v1.image.decode_jpeg(image)
  image = tf.expand_dims(image, axis=0)
  # Resize and pad the image to keep the aspect ratio and fit the expected size.
  image = tf.cast(tf.image.resize_with_pad(image, 256, 256), dtype=tf.int32)

  # Download the model from TF Hub.
  model = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
  movenet = model.signatures['serving_default']

  # Run model inference.
  outputs = movenet(image)
  # Output is a [1, 1, 17, 3] tensor.
  keypoints = outputs['output_0']
 ```
 ## Example on a real Dataset
 You can download the yoga-pose dataset from [here](https://laurencemoroney.com/2021/08/23/yogapose-dataset.html)
![image](https://user-images.githubusercontent.com/47725118/130782594-675721c3-e6a8-417d-b963-73d001400993.png)

  
