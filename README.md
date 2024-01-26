# Simple Fall Detection
My internship project/mini-prototype at [Intelligent Design Technology Limited](https://intelligentdesign.hk/english/) that aims to detect in real-time whether a person is falling, sitting, standing, or walking using only a camera image input.

<p align="middle">
    <img src="./docs/media/demo-1.gif" width="30%" height="200px"/>
    <img src="./docs/media/demo-2.gif" width="30%" height="200px"/>
    <img src="./docs/media/demo-3.gif" width="30%" height="200px"/>
</p>

## About
This project is done using pose estimation and basic heuristics. This project is designed for an elderly home monitoring system that uses Raspberry PI. Tested with both Yolo V8 pose and Movenet pose models.



## Usage
### Install Dependencies
- Make sure to have Python 3 and Pip. 
- (Optional) Make and activate Python virtual environment

``` bash
python -m venv venv
source venv/bin/activate
```

- Install requirements:

``` bash
pip install ultralytics
```

- Clone Repo
```bash
git clone https://github.com/jamesyoung-15/Fall-Detection
```

### Example Run
-  Yolo Pose Estimate on Video
```bash
python run.py --src test/data/videos/fall-1.mp4 --interval 7
```

-  Yolo Pose Estimate on USB-Camera
```bash
python run.py --src /dev/video0 --interval 7 --width 640 --height 480
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

## Documentation
For full documentation see [here.](./docs/Documentation.md)
### Possible Approaches to Fall Detection/Action Detection
1. Use pose estimation w/ heuristics

2. Video classification

3.  Image Classification

4. Combine some of the above methods (eg. pose estimate + classification).


## Resources
- [Yolo Pose](https://docs.ultralytics.com/tasks/pose/)
- [Movenet Example](https://github.com/Kazuhito00/MoveNet-Python-Example/tree/main)
- [Example of using Pose to determine posture](https://bozliu.medium.com/dynamic-siting-posture-recognition-and-correction-68ae418fbc77)
- [Example of using OpenPose to determine fall w/ head](https://github.com/augmentedstartups/Pose-Estimation/tree/master/3.%20Fall%20Detection)