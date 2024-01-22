# Notes
My notes.

## Progress
### Completed
- Setup Pose detection for MediaPipe (Blazenet), Movenet (multi and single), and Yolo Pose
- Setup faster usb camera streaming with threading
- Managed to extract pose keypoints and use them to make basic sit, stand, lie down

### Todo
- Improve heuristic
  - false positive when person is bending down but not falling (ie. touching toes)
- Further optimize
  - still less than 1 fps on RPI 4B, need to optimize (eg. multi-threading)


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

## Yolo Pose Estimate Approach

- Misclassification from 50 Ways to Sit:
  - Tying shoe
  - silent film star
- Misclassification from 50 Ways to Jump:
  - todo
- Misclassification from 50 Ways to Fall:
  - todo

- Le2i
  - Coffee Room 01
    - Incorrect: 5 8 14 15 26 29 30 33 35 36 37 44
    - Accuracy: 0.75
  - Coffee Room 02
    - Incorrect: 54 57 59 62
    - Accuracy: 0.8

## Datasets
  - [small kaggle dataset w/ 3 classes (sit, stand, fall)](https://www.kaggle.com/datasets/uttejkumarkandagatla/fall-detection-dataset)
  - [medium fall dataset w/ 1 class (fall detected)](https://universe.roboflow.com/roboflow-universe-projects/fall-detection-ca3o8)
  - [Le2i Fall Videos](https://www.kaggle.com/datasets/tuyenldvn/falldataset-imvia)
  - [Fall videos](https://kuleuven.app.box.com/s/dyo66et36l2lqvl19i9i7p66761sy0s6)

## Quick Notes
- On RPI Yolo pose getting >1 fps and is extremely choppy, need to further optimize
- Movenet with TFlite is faster but pretty inaccurate

- Improvements:
    - Mike said that using image classification better than pose estimate, perhaps I'll try both
    - Vincent said rather than combining inference and camera stream in one task, can split into two 
        - If inference takes too long, can put pending