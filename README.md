# Simple Fall Detection
<!-- Github Repo [here](https://github.com/jamesyoung-15/FallDetection). Quick proto-type project that aims to detect in real-time whether a person is falling, sitting, standing, or walking using only a camera image input. -->

My internship project/mini-prototype at [Intelligent Design Technology Limited](https://intelligentdesign.hk/english/) that aims to detect in real-time whether a person is falling, sitting, standing, or walking using only a camera image input.

<p align="middle">
    <img src="./docs/media/demo-1.gif" width="30%" height="200px"/>
    <img src="./docs/media/demo-2.gif" width="30%" height="200px"/>
    <img src="./docs/media/demo-3.gif" width="30%" height="200px"/>
</p>

## About
This project has two implementations. The first implementatnion uses pose detection, where it uses pose estimation to extract a person's skeleton and uses heuristics (ie. angle between certain body parts) to determine a person's action. The second implementation uses an image classification to determine the person's state. Both of these approaches are currently not robust enough and need improvements, but can do some basic detection.



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

### Example Run w/ Pose Approach
-  Yolo Pose Estimate on Video
```bash
python pose_fall_demo.py --src test/data/videos/fall-1.mp4 --interval 5 --pose_type 0
```

-  Yolo Pose Estimate on USB-Camera
```bash
python run.py --src /dev/video0 --interval 7 --width 480 --height 320
```


## Documentation
For full documentation see [here.](./docs/Documentation.md)
### Possible Approaches to Fall Detection/Action Detection
1. Use pose estimation w/ heuristics

2. Video classification

3. Image Classification

4. Combine some of the above methods (eg. pose estimate + classification).

In the end I could only implement 1 and 3. 

## Resources
- [Yolo Pose](https://docs.ultralytics.com/tasks/pose/)
- [Movenet Example](https://github.com/Kazuhito00/MoveNet-Python-Example/tree/main)
- [Example of using Pose to determine posture](https://bozliu.medium.com/dynamic-siting-posture-recognition-and-correction-68ae418fbc77)
- [Example of using OpenPose to determine fall w/ head](https://github.com/augmentedstartups/Pose-Estimation/tree/master/3.%20Fall%20Detection)