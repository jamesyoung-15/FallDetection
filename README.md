# Fall Detection
Attempt at fall detection using pose estimation. Project aims to detect in real-time whether a person is falling, sitting, standing, or moving/walking using only a camera input. Built for Raspberry Pi w/ camera.

## Comparing Pose Estimation Models
There are many pose detection models available. Below are some I considered and tested:
- Movenet
- Posenet
- Blazenet (from Mediapipe)
- Yolo V8 Pose

Seems like Movenet or Yolo V8 most suitable as it supports multi-person pose detection. Need further testing for performance and speed.

### Movenet

|Id |	Part|
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
|16| 	rightAnkle