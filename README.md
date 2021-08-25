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
MobileNetV2 architecture : 
![image](https://user-images.githubusercontent.com/47725118/130789918-65d968b8-5a5e-46a2-9129-8d9225435eb6.png)
### Inputs
A frame of video or an image, represented as an int32 tensor of shape: 192x192x3(Lightning) / 256x256x3(Thunder). Channels order: RGB with values in [0, 255].
### Outputs
A float32 tensor of shape [1, 1, 17, 3].
● The first two channels of the last dimension represents the yx coordinates (normalized to image frame, i.e. range in [0.0, 1.0]) of the 17 keypoints (in the order of: [nose, left eye, right eye, left ear, right ear, left shoulder, right shoulder, left elbow, right elbow, left wrist,right wrist, left hip, right hip, left knee, right knee, left ankle, right ankle]).
● The third channel of the last dimension represents the prediction confidence scores of
each keypoint, also in the range [0.0, 1.0].
![image](https://user-images.githubusercontent.com/47725118/130782594-675721c3-e6a8-417d-b963-73d001400993.png)

  
