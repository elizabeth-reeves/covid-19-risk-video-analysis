# Covid-19 Risk Video Analysis
This is an integration of a social distancing algorithm, a face mask detecting algorithm, and a person detecting algorithm. It is designed to track the number of people who pass through a video clip, including whether or not they are wearing masks, and whether or not they are social distancing. It will annotate the video frame by frame with this analysis, and can display the annotation live, or write it to an output file .avi file. Finally, when the video stream ends, it will write a csv of data to a specified output file location, including time series data for all frames of the video, and a calculated Risk Index between 0 and 100.

## Requirements
Found in requirements.txt. User must install:
- tensorflow>=1.15.2
- keras>=2.3.1
- imutils>=0.5.3
- numpy>=1.18.2
- opencv-python>=4.2.0.*
- matplotlib>=3.2.1
- argparse>=1.1
- scipy>=1.4.1
- scikit-learn>=0.23.1
- pillow>=7.2.0

## Methodology

## Usage:
The main function called is facemask_tracking.py. This file optionally takes as input a prerecorded video, or a camera id; if no video or camera id is provided as input, it will by default run on the video stream from the default camera of the device. 

- "--video" or "-v" is a flag available to specify an input source for video. It can be passed "0" to use the default camera for video streaming, a different camera id if known, or a relative or absolute path to a prerecorded video. Extensions supported include .avi, .mp4, and .mov
```bash
python3 facemask_tracking.py -v 0
python3 facemask_tracking.py --video ./input_images_and_videos/pedestrian_survaillance.mp4
```

- "--roi" or "-r" is a flag available to specify the relative position of the ROI (region of interest) within the frames being analyzed. By default, it is 0.5, indicating that the ROI will be a vertical line in the middle of the frame. This should be a value between 0 and 1.
```bash
python3 facemask_tracking.py --roi 0.25
python3 facemask_tracking.py -v 0 -r 0.8
```

- "--deviation" or "-d" is a flag available to specify the width of the detection area around the ROI. By default, it is 10 pixels.
```bash
python3 facemask_tracking.py --deviation 10
python3 facemask_tracking.py -r 0.4 -d 15
```

## Tuning the detection algorithms
There is a tradeoff between detecting a single person more than once (multiple detections) and not detecting a person at all (zero detections). This tradeoff relies on the fps (frames per second) of the video, and on the value of --deviation. The larger --deviation is, the more likely we are to count a single person multiple times, and the less likely we are to miss anyone. With higher fps, it is possible to decrease the value of the deviation, decreasing the area of detection, and thus reduce the number of mulitple-counts we get, without missing individuals.


## Demo:
![gif1](./embedded/demoGif.gif)


## Face Mask Model Sourced From:
[Face-Mask-Detection](https://github.com/chandrikadeb7/Face-Mask-Detection) by [Chandrika Deb](https://github.com/chandrikadeb7), 2020.

## Base Object Detection API Sourced From:
[TensorFlow Object Counting API](https://github.com/ahmetozlu/tensorflow_object_counting_api) by [Ahmet Özlü](https://github.com/ahmetozlu), 2018.
