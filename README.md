# Simple Action Detection
Project aims to detect in real-time whether a person is falling, sitting, standing, or walking using only a camera input. Built for Raspberry Pi w/ camera.

## Usage
Todo

## Approaches
1. Use pose estimation w/ heuristics
    - pros:
        - calculate angle between body parts to determine action
        - relatively simple compared to below methods
    - cons:
        - computationally expensive, especially w/ multiple people
        - need good heuristic, not easy (many edge cases/weird poses)

2. Use action recognition model (video classification). Could use pre-trained or train own model w/ transfer learning.
    - pros:
        - could use transfer learning with model like movinet (compatible with tflite)
    - cons:
        - downside is less tutorials/guides on how to implement and harder to find dataset 
        - not sure how performance will be on RPI

3. Train an image classifier w/ dataset.
    - pros:
        - lots of tutorials/guides on image classification, can use many models for transfer learning
        - should be more faster than pose estimation especially when there are multiple people
    - cons: 
        - problem is not many public/reputable dataset, won't be robust

4. Combine some of the above methods (eg. pose estimate + classification).

### Comparing Pose Estimation Models
There are many pose detection models available. Below are some I considered and tested|
- Movenet
- Posenet
- Blazenet (from Mediapipe)
- Yolo V8 Pose

Seems like Movenet or Yolo V8 most suitable as it supports multi-person pose detection. Need further testing for performance and speed.

#### Movenet and Yolo Pose Keypoints

<!-- |Id |	Part|
|---|---|
|0|	nose|
|1| 	leftEye
|2| 	rightEye
|3| 	leftEar
|4| 	rightEar
|5| 	leftShoulder
|6| 	rightShoulder
|7| 	leftElbow
|8| 	rightElbow
|9| 	leftWrist
|10| 	rightWrist
|11| 	leftHip
|12| 	rightHip
|13| 	leftKnee
|14| 	rightKnee
|15| 	leftAnkle
|16| 	rightAnkle -->

|Part|ID|
|-|-|
|NOSE|           0|
|LEFT_EYE|       1|
|RIGHT_EYE|      2|
|LEFT_EAR|       3|
|RIGHT_EAR|      4|
|LEFT_SHOULDER|  5|
|RIGHT_SHOULDER| 6|
|LEFT_ELBOW|     7|
|RIGHT_ELBOW|    8|
|LEFT_WRIST|     9|
|RIGHT_WRIST|    10|
|LEFT_HIP|       11|
|RIGHT_HIP|      12|
|LEFT_KNEE|      13|
|RIGHT_KNEE|     14|
|LEFT_ANKLE|     15|
|RIGHT_ANKLE|    16|


## Resources
(Movenet Example)[https://github.com/Kazuhito00/MoveNet-Python-Example/tree/main]
(Potential way to increase USB camera FPS)[https://pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/]