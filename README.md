Sample Video- https://www.youtube.com/watch?v=djHRAaRSIzs


This projects couts the repetitions of common exercises. Here I have provided support for pushups, chinups and squats. It can also tell whether you are doing a complete rep or an incomplete repetition. It can be extended to other exercises as well and they can be auto detected by just noting the angles between different joints.
This projects counts the repetitions of common exercises. Here I have provided support for pushups, chinups, squats, sidecurl and dumbell curl. It can also tell whether you are doing a complete rep or an incomplete repetition. It can be extended to other exercises as well and they can be auto detected by just noting the angles between different joints.
The model is also scalable to multi person estimation.

Here I used pose_hrnet_w48_256*192 pretrained on COCO dataset with 17 joints. 
The model works irrespective of the direction of the camera except from the back side. 
The model works irrespective of the direction of the camera. 
For PushUps I used the angle formed by the elbow, for Squats the angle formed by the knees was used and distance between nose and wrists was considered for chin ups.


This works only with video


### Run

For running the web application go to [http://142.93.222.67/](http://142.93.222.67/). 
1-Choose the file for which you need to count the reps<br/>
2-Choose the type of exercise.<br/>
3-Enter the mailId on which you want to receive the final video.<br/>

The video takes some time to process, so wait for 5-10mins for the mail. Mail might get into spam so check it after 10mins. Video cannot be viewed on the browser, download the video from the link.
<br/>

For test purposes you can download a sample video from [google drive](https://drive.google.com/drive/folders/1GDE8TySO5LBN6doJtW9DvAbtu-av1ivI?usp=sharing). 

<br/>
To run the application on local system, run start-count.py and give filename with type of exercise to be counted.(1 for pushUps, 2 for sitUps, 3 for chinUps). 
To run the application run start-count.py and give filename with type of exercise to be counted.
1-PushUps<br/>
2-SitUps<br/>
3-ChinUps<br/>
4-Dumbell Curl<br/>
5-Side Dumbell Lateral<br/>

For running the application you need to add the weights folder in the main directory which can be downloaded from [google drive](https://drive.google.com/drive/folders/1GDE8TySO5LBN6doJtW9DvAbtu-av1ivI?usp=sharing). 

<br/>

For eg.  ```python main.py --filename test.mp4 --exercise_type 1 ```

### examples

Sample input video and weights for the pipeline can be found at 

<img src="https://github.com/akshatkaush/exercise-count/blob/master/New%20folder/chinups_sample.PNG?raw=true"  >

<img src="https://github.com/akshatkaush/exercise-count/blob/master/New%20folder/push_up_sample.PNG?raw=true" width="568.5" height="286.5">

<img src="https://github.com/akshatkaush/exercise-count/blob/master/New%20folder/push_up_sample2.PNG?raw=true" width="568.5" height="286.5">

<img src="https://github.com/akshatkaush/exercise-count/blob/master/New%20folder/frame.png?raw=true">

<img src="https://github.com/akshatkaush/exercise-count/blob/master/New%20folder/websample.PNG?raw=true" width="568.5" height="286.5">

