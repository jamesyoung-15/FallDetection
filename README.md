# Simple Action Detection
My internship project/mini-prototype that aims to detect in real-time whether a person is falling, sitting, standing, or walking using only a camera input. Prototype built for Raspberry Pi w/ camera.

## Usage
### Install Dependencies
- Make sure to have Python 3 and Pip. Make Python virtual env and install requirements:
``` bash
pip install ultralytics

```

### Example Run
-  Yolo Pose Estimate on Video
```bash
python run.py --src test/data/videos/fall-1.mp4
```

- Yolo Pose Estimate on Image
```bash
python run.py --src test/data/images/lie-down-1.mp4 --type 1
```


## Possible Approaches
1. Use pose estimation w/ heuristics

2. Use action recognition model (video classification). Could use pre-trained or train own model w/ transfer learning.

3. Train an image classifier w/ dataset.

4. Combine some of the above methods (eg. pose estimate + classification).

### Comparing Pose Estimation Models
There are many pose detection models available. Below are some I considered:
- Movenet
- Posenet
- Blazenet (from Mediapipe)
- Yolo V8 Pose

Seems like Movenet or Yolo V8 most suitable as it supports multi-person pose detection. Movenet multi-pose lite is faster (~ 4 fps on RPI 4) but Yolo is more accurate. Will probably use Yolo and try to optimize further. 

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
(Yolo Pose)[https://docs.ultralytics.com/tasks/pose/]
(Movenet Example)[https://github.com/Kazuhito00/MoveNet-Python-Example/tree/main]
(Potential way to increase USB camera FPS)[https://pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/]
(Example of using Pose to determine posture)[https://bozliu.medium.com/dynamic-siting-posture-recognition-and-correction-68ae418fbc77]