# Covid-19 Risk Video Analysis
This is an integration of a social distancing algorithm, a face mask detecting algorithm, and a person detecting algorithm. It is designed to track the number of people who pass through a video clip, including whether or not they are wearing masks, and whether or not they are social distancing. It will annotate the video frame by frame with this analysis. Finally, at the end of the clip, it will output a Risk Index, based on these variables, as well as time series data for the variables tracked.

## Requirements
Found in requirements.txt. User must install:
tensorflow>=1.15.2
keras==2.3.1
imutils==0.5.3
numpy==1.18.2
opencv-python==4.2.0.*
matplotlib==3.2.1
argparse==1.1
scipy==1.4.1
scikit-learn==0.23.1
pillow==7.2.0
streamlit

## Usage:
Analysis of the pedestrian_survaillance.mp4 video
```bash
python3 facemask_tracking.py --video ./input_images_and_videos/pedestrian_survaillance.mp4
```

Analysis of video streamed from a webcam (default camera for the computer it's run on)
```bash
python3 facemask_tracking.py
```
