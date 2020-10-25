# exercise-count
Here I used pose_hrnet_w48_384x288 pretrained on COCO dataset with 17 joints. 
The model works irrespective of the direction of the camera except from the back side. 
For PushUps I used the angle formed by the elbow, for Squats the angle formed by the knees was used and distance between nose and wrists was considered for chin ups.

It can be extended to other exercises as well and they can be auto detected by just noting the angles between different joints.
The model is also scalable to multi person estimation.

To run the application run start-count.py and give filename with type of exercise to be counted.(1 for pushUps, 2 for sitUps, 3 for chinUps)
For eg. python startcount.py --filename test.mp4 --exercise_type 1

Sample Video- https://www.youtube.com/watch?v=djHRAaRSIzs
