# Documentation
Work documentation.

## List of Useful Datasets
- [small kaggle dataset w/ 3 classes (sit, stand, fall)](https://www.kaggle.com/datasets/uttejkumarkandagatla/fall-detection-dataset)
- [Roboflow image dataset w/ 4 classes (fall, sit, stand, walk)](https://universe.roboflow.com/customdataset-lmry5/human-fall-detection-hdkty/dataset/8)
- [Le2i Fall Videos](https://www.kaggle.com/datasets/tuyenldvn/falldataset-imvia)
- [Fall videos](https://kuleuven.app.box.com/s/dyo66et36l2lqvl19i9i7p66761sy0s6)
- [Fall videos](http://www.iro.umontreal.ca/~labimage/Dataset/)

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

- With USB-Camera Example (limiting usb-camera fps if possible)

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
The files/code are in `Pose_Estimate` and `utils` folder. See `pose_fall_detection.py` and `pose_utils.py` for the heuristic algorithm (just a bunch of random angles between joints and vectors).

### State Recognition (Sit, Stand, Lying Down)
See `pose_utils.py` file. The function `determine_state` shows the implementation. It just uses a bunch of random angles between joints and vectors (eg. spine and leg parallel to ground likely means lying down, etc).

### Fall Detection
The idea is taken from the project [ambianic](https://ambianic.ai/) which also uses pose detection (PoseNet) and calculates the change in spine vector for determining fall. See below for basic idea of how they did it:

![](https://user-images.githubusercontent.com/2234901/112545190-ea89d380-8d85-11eb-8e2c-7a6b104d159e.png)

My implementation is similar, see `pose_fall_detection.py`. There are 3 frames of pose detections and states that are stored in a dictionary called `prev_data`. Then we calculate the angles between the spine vector and the change in y-coordinates of the hip and shoulders. 

The idea is that hip and shoulder y-change means the person's body is dropping and a change in spine vector means the body is most likely falling. The reason to include the hip and shoulder y-change is to avoid situations where a person is bending down to reach something to count as fall, as bending down usually means the hip won't drop down.

### Comparing Pose Estimation Models
There are many pose detection models available. Below are some I considered:
- Movenet
- Posenet
- Blazenet (from Mediapipe)
- Yolo V8 Pose
- OpenPose

I chose to use Yolo Pose and Movenet Pose as they support multi-person detection and also are relatively light-weight (can run on Raspberry PI).

#### Movenet and Yolo Pose Keypoints Mapping
![](./media/images/yolo-pose-keypoints.png)

### Problems
The tracking algorithms for giving unique IDs to each person don't always work properly (ie. sometimes single person can be given multiple unique IDs, meaning that the fall won't be detected). 

This tracking problem exist for both Yolo and Movenet models.Therefore perhaps instead of using multi-person pose model can use single-pose model instead (but this means it may not work when more than one person is in frame).

Also the pose models doesn't always detect a person, especially in low-lighting situations or if a person is far away.

Another issue is the heuristic algorithm needs more testing and can sometimes give false positives. Because the algorithm calculates the change in spine vector as well as the hips and shoulder y-coordinate change across 3 frames, this can still sometime detect falls when there isn't. For example, if a person goes from standing to praying on his knees (ie. Muslim praying, see `test-data/videos/nofall/nofall-6.mp4`), this may detect a fall when there isn't.

As you can imagine, using heuristics isn't the best approach but with the hardware limitation it is difficult to add something like a pose action classification as the pose inference itself is already computationally expensive for the Raspberry PI.

### Improvements

If possible, I would recommend trying to approach the project like this -> [Human-Falling-Detect-Tracks](https://github.com/GajuuzZ/Human-Falling-Detect-Tracks), where on top of pose detection, you add a pose-based action classifier. However, this project cannot run on Raspberry PI (got less than 1 fps).

Alternatively use image classificatin (see below) or train a video classification (like [Movinet](https://www.tensorflow.org/hub/tutorials/movinet)) with video dataset.

## How it works (Image Detection Approach)
This approach detects 4 actions (falling, sitting, standing, walking). It performs action recognition on single frame (image classification) rather than other approaches like video classification that takes multiple frames. 

I used [this](https://universe.roboflow.com/customdataset-lmry5/human-fall-detection-hdkty/dataset/8) image dataset. Used transfer learning starting with Yolo V8's `yolov8n.pt` model. See `image_based` folder for the training script and `image_fall_demo.py` for inference script.

### Problems
This approach detects many false positives (detecting falls when there isn't even a person in frame). This is most likely due to the small dataset size and/or the dataset used isn't good enough (ie. not diverse enough, etc). 

### Improvements
- Can use a better dataset (add more images or use another one).
- Better image pre-processing to avoid false detections (eg. only perform classification if person detected)
- Can also use another approach, use video classification instead of image classification (see action recognition using video classfication)