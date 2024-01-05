# Notes
My notes.

## Progress

- [] 
- []
- 


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

## Quick Notes
- Mike said that using image classification better than pose estimate
- Vincent said rather than combining inference and camera stream in one task, can split into two 
    - If inference takes too long, can put pending