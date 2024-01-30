# Documentation
Work documentation.

## Setup
My project was tested on Linux (Pop-OS 22.04) w/ Python 3.10.12.
### Dependencies
- Python 3
- Pip

### Main Packages
- Ultralytics (should contain all other needed packages like cv2, tensorflow, etc.)

### Setup
- Clone and Enter Repo:

``` bash
git clone https://github.com/jamesyoung-15/FallDetection
cd FallDetection
```

- (Optional) Make and activate Python virtual environment

``` bash
python -m venv venv
source venv/bin/activate
```

- Install Required Packages:

``` bash
pip install ultralytics
```

## Usage
### Example Run w/ Pose
If CV2 shows video running too fast, can set `--delay 20` to decrease fps.

- With Yolo Pose

``` bash
python pose_fall_demo.py --src test-data/videos/fall/fall-1.mp4 --interval 5
```

- With Movenet

``` bash
python pose_fall_demo.py --src test-data/videos/fall/fall-1.mp4 --pose_type 1
```

- With USB-Camera Example

``` bash
python pose_fall_demo.py --src /dev/video0 --pose_type 1 --fps 10
```

Full Command-Line Options
``` bash
  -h, --help            show this help message and exit
  --src SRC             Video file location (eg. /dev/video0)
  --show SHOW           Whether to show camera to screen, 0 to hide 1 to show.
  --width WIDTH         Input video width. (eg. 640)
  --height HEIGHT       Input video height (eg. 480)
  --conf_score CONF_SCORE
                        Confidence score threshold (eg. 0.7)
  --interval INTERVAL   Interval in frames to run inference (eg. 2 means inference every 2 frames)
  --manual_frame MANUAL_FRAME
                        Set this to 1 if you want to press 'n' key to advance each video frame.
  --type TYPE           Specifies whether input is image or video (0 for video 1 for image). Default is video (0).
  --debug DEBUG         Whether to print some debug info. Default is 0 (no debug info), 1 means print debug info.
  --save_vid SAVE_VID   Whether to save video. Default is 0 (no save), 1 means save video.
  --pose_type POSE_TYPE
                        Specify which pose model to use. 0 for YoloV8Pose (default), 1 for Movenet Multi Lightning.
  --resize_frame RESIZE_FRAME
                        Whether to resize frame. Default is 0 (no resize), 1 means resize frame.
  --delay DELAY         Delay in ms for cv2.waitkey(delay) in int.
  --fps FPS             Set FPS for cv2 (only for usb camera).
  --benchmark BENCHMARK
                        Record and print FPS after exit. 0 for false 1 for true.
```

### Example Run w/ Image Classification Approach

``` bash
python image_fall_demo.py --src test-data/videos/fall/fall-1.mp4 
```


## How it works (Pose)
The files/code are in `Pose_Estimate` and `utils` folder. See `pose_fall_detection.py` and `pose_utils.py` for the heuristic algorithm (probably not best approach just a bunch of random angles between joints and vectors).

### Comparing Pose Estimation Models
There are many pose detection models available. Below are some I considered:
- Movenet
- Posenet
- Blazenet (from Mediapipe)
- Yolo V8 Pose
- OpenPose

I chose to use Yolo Pose and Movenet Pose as they support multi-person detection and also are relatively light-weight (can run on Raspberry PI).

#### Movenet and Yolo Pose Keypoints Mapping
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

![](./media/images/yolo-pose-keypoints.png)

### Problems
The tracking algorithms for giving unique IDs to each person don't always work properly (ie. sometimes single person can be given multiple unique IDs, meaning that the fall won't be detected). Therefore perhaps instead of using multi-person pose model can use single-pose model instead.

Also the pose models doesn't always detect a person, especially in low-lighting situations or if a person is far away.

Another issue is the heuristic algorithm needs more testing and can sometimes give false positives. Because the algorithm calculates the change in spine vector as well as the hips and shoulder y-coordinate change across 3 frames, this can still sometime detect falls when there isn't. For example, if a person goes from standing to praying on his knees (ie. see Muslim praying), this may detect a fall when there isn't.

## How it works (Image Detection)
This approach detects 4 actions (falling, sitting, standing, walking). It performs action recognition on single frame (image classification) rather than other approaches like video classification that takes multiple frames. 

I used [this](https://universe.roboflow.com/customdataset-lmry5/human-fall-detection-hdkty/dataset/8) image dataset. Used transfer learning starting with Yolo V8's `yolov8n.pt` model. See `image_based` folder for the training script and `image_fall_demo.py` for inference script.

### Problems
This approach detects many false positives (detecting falls when there is nothing). This is most likely due to the small dataset size and/or the dataset used isn't good enough (ie. not diverse enough, etc). 