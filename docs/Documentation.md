# Documentation
My work documentation.

## Setup
My project was tested on Linux (Pop-OS 22.04) w/ Python 3.10.12.
### Dependencies
- Python 3
- Pip

### Main Packages
- Yolo V8
- OpenCV (included in Yolo pip install)

### Setup
- Clone and Enter Repo:

``` bash
git clone https://github.com/jamesyoung-15/Fall-Detection
cd Fall-Detection
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
### Example Run
-  Yolo Pose Estimate on Video
```bash
python run.py --src test/data/videos/fall-1.mp4 --interval 7
```

-  Yolo Pose Estimate on USB-Camera
```bash
python run.py --src /dev/video0 --interval 7 --width 480 --height 480
```

- Yolo Pose Estimate on Image
```bash
python run.py --src test/data/images/lie-down-1.mp4 --type 1
```

Command-Line Options:
``` bash
options:
  -h, --help                    show this help message and exit
  --src SRC                     Video file location (eg. ./test-data/videos/fall-1.mp4)
  --show SHOW                   Whether to show camera to screen, 0 to hide 1 to show.
  --width WIDTH                 Input video width. (eg. 480)
  --height HEIGHT               Input video height (eg. 480)
  --conf_score CONF_SCORE       Confidence score threshold (eg. 0.7)
  --interval INTERVAL           Interval in seconds to run inference (eg. 2)
  --manual_frame MANUAL_FRAME   Set this to 1 if you want to press 'n' key to advance each video frame.
  --type TYPE                   Specifies whether input is image or video (0 for video 1 for image). Default is video (0).
  --debug DEBUG                 Whether to print some debug info. Default is 0 (no debug info), 1 means print debug info.
```



## How it works

### Comparing Pose Estimation Models
There are many pose detection models available. Below are some I considered:
- Movenet
- Posenet
- Blazenet (from Mediapipe)
- Yolo V8 Pose

Seems like Movenet or Yolo V8 most suitable as it supports multi-person pose detection. Movenet multi-pose lite is faster but Yolo is more accurate. Will probably use Yolo and try to optimize further. 

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

## Improvements
