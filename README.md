# Human Pose Estimation and Comparison

Install Dependencies:
```
pip install -r requirements.txt
```
There aree some sample test videos named ```testing.mp4```, ```testing2.mp4```  
The file called ```compare_ground_truth.pickle``` contains the sequence of keypoints recorded for specific action recorded from test video.

In order to compare ```testing.mp4``` with the keypoints recorded in the ```compare_ground_truth.pickle``` under the label ```action```,run:
```
python main.py --activity "body_exercise" --video "testing.mp4"
```
## Creating New Lookup

There is a file ```fetch_keypoints.py``` which can be used to create a new lookup table. In order to extract and record keypoints from ```ground_truth.mp4```, run:
```
python fetch_keypoints.py --activity "body_exercise" --video "ground_truth.mp4" --lookup "lookup_new.pickle"/[YOUR_LOOKUP_NAME]
```
Then, in order to use this new lookup, run:
```
python main.py --activity "body_exercise" --video "ground_truth.mp4" --lookup "lookup_new.pickle"/[YOUR_LOOKUP_NAME]
```
