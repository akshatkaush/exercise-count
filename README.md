# exercise-count
This projects couts the repetitions of common exercises. Here I have provided support for pushups, chinups and squats. It can also tell whether you are doing a complete rep or an incomplete repetition. It can be extended to other exercises as well and they can be auto detected by just noting the angles between different joints.
The model is also scalable to multi person estimation.

Here I used pose_hrnet_w48_384x288 pretrained on COCO dataset with 17 joints. 
The model works irrespective of the direction of the camera except from the back side. 
For PushUps I used the angle formed by the elbow, for Squats the angle formed by the knees was used and distance between nose and wrists was considered for chin ups.



This works only with video


### Run 
To run the application run start-count.py and give filename with type of exercise to be counted.(1 for pushUps, 2 for sitUps, 3 for chinUps). 

For eg.  ```python main.py --filename test.mp4 --exercise_type 1 ```


Sample Video- https://www.youtube.com/watch?v=djHRAaRSIzs

### examples

<img src="https://github.com/akshatkaush/exercise-count/blob/master/New%20folder/chinups_sample.PNG?raw=true"  >

<img src="https://github.com/akshatkaush/exercise-count/blob/master/New%20folder/push_up_sample.PNG?raw=true" width="568.5" height="286.5">

<img src="https://github.com/akshatkaush/exercise-count/blob/master/New%20folder/push_up_sample2.PNG?raw=true" width="568.5" height="286.5">

<img src="https://github.com/akshatkaush/exercise-count/blob/master/New%20folder/squatssample.PNG?raw=true">




