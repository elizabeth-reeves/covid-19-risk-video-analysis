import numpy as np
import argparse
import sys
import cv2
from math import pow, sqrt
import imageio
from typing import Tuple, Dict, List


# Parse the arguments from command line
arg = argparse.ArgumentParser(description='Social distance detection')

arg.add_argument('-v', '--video', type = str, default = '', help = 'Video file path. If no path is given, video is captured using device.')

arg.add_argument('-m', '--model', required = True, help = "Path to the pretrained model.")

arg.add_argument('-p', '--prototxt', required = True, help = 'Prototxt of the model.')

arg.add_argument('-l', '--labels', required = True, help = 'Labels of the dataset.') # our model will have face labels and maybe bus depth label

arg.add_argument('-c', '--confidence', type = float, default = 0.8, help='Set confidence for detecting objects')

arg.add_argument('-F', '--focal', type = int, default = 904, help='Adjust Focal Length of Camera')

arg.add_argument('-r', '--record', type = bool, default = False, help='To record or not to record, that is the question')



args = vars(arg.parse_args())


labels = [line.strip() for line in open(args['labels'])]


# Load model
print("\nLoading model...\n")
network = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

print("\nStreaming video using device...\n")


def gif_to_imgs(fname):
    gif = imageio.mimread(fname)
    n_frames = len(gif)
    imgs = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in gif]
    return n_frames, imgs

gif = 0

# Capture video from file or through device
if args['video']:

    video_file = args['video']
    if video_file[-3:] == 'gif':
        n_frames, imgs = gif_to_imgs(video_file)
        gif = 1

    
    cap = cv2.VideoCapture(args['video']) # stream a video
else:
    cap = cv2.VideoCapture(0) # stream webcam


# if recording is happening
if args['record']:
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')

    if not gif:
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))

    else:
        frame_width, frame_height, channels = imgs[0].shape 

    fps = 20
    if gif:
        out = cv2.VideoWriter('output.avi', fourcc, fps, (frame_height, frame_width))
    else:
        out = cv2.VideoWriter('output.avi', fourcc, fps, (frame_width, frame_height))



# convert cm to ft
def cm_to_ft(cm: float) -> float:
    return round( cm / 30.48, 4)



# Assume the "raw"/actual height of an average person is 165 cm.
raw_height = 165 # cm.

# Focal length (this is a FIXED variable that **variates per camera or video.** Adjust it to your own)
#F = 615
F = args['focal']

def find_raw_coords(idx: int, tl: Tuple[int, int], br: Tuple[int, int ]) -> None:
    '''
    Calculate and store raw x,y,z coordinates of each ROI in cm.

    NOTE: vars "F", "raw_height", and "raw_ROI_coords" must be defined outside of function before being called
    '''

    # unpack coordinates
    tl_x, tl_y = tl

    br_x, br_y = br


    # Mid point of bounding box
    x_mid = round((tl_x + br_x)/2,4)
    y_mid = round((tl_y + br_y)/2,4)

    pixel_center_ROI_coords[idx] = (x_mid, y_mid)

    # pixel height of our detected ROI
    pixel_height = round(tl_y - br_y,4)
        
    # Distance from ROI to camera based on triangle similarity (calculating depth)
    depth_cm = (raw_height * F)/ pixel_height # cm.


    # estimate the raw mid-point dimensions of of our bbox (in cm) based on triangle similarity tech.

    x_mid_cm = (x_mid * depth_cm) / F # raw dimension of single pixel x
    y_mid_cm = (y_mid * depth_cm) / F # raw dimension of single pixel y

    # store the raw 3d coordinates of the center of our ROI 
    raw_ROI_coords[idx] = (x_mid_cm,y_mid_cm, depth_cm)

    return None




def are_ROIs_close(idx_i: int, idx_j: int) -> None:
    '''
    Computes the distance between ROI i and j.

    NOTE: vars "close_ROI_pairs" and "close_ROI_dist" must be defined outside of function before being called
    '''

    # compute raw (in cm.) 3d distance between two objects using Euclidean distance
    x_dist = pow(raw_ROI_coords[idx_i][0]-raw_ROI_coords[idx_j][0],2)
    y_dist = pow(raw_ROI_coords[idx_i][1]-raw_ROI_coords[idx_j][1],2)
    z_dist = pow(raw_ROI_coords[idx_i][2]-raw_ROI_coords[idx_j][2],2)
    
    dist = sqrt(x_dist + y_dist + z_dist)

    # Check if distance less than 2 metres or 200 centimetres ( 6ft == 182.88 cm)
    
    # if True, ROI objects are NOT maintaining social distancing
    if dist < 200:

        key = (idx_i,idx_j)
        
        # store pair of indices
        close_ROI_pairs.append( key )

        # store distance of pair indices
        close_ROI_dist[key] = dist # cm

    return None

    

# record total number of frames
frame_no = 0


while cap.isOpened():

    if gif:
        if frame_no + 1 > n_frames:
            break
        frame = imgs[frame_no]



    # frame_no will count total number of frames taken in our video camera
    frame_no = frame_no+1


    # Capture one frame after another
    # ret is boolean. If True, it essentially means the picture was read correctly

    if not gif:
        ret, frame = cap.read()

        if not ret:
            break


    (h, w) = frame.shape[:2] # we only want height and width, not n_channels

    # Resize the frame to suite the model requirements. Resize the frame to 300X300 pixels
    # below performs: mean subtraction, scaling, and informs the dim that the NN expects
    # essentially, it normalizes our image
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 
            scalefactor = 0.007843, size = (300, 300), mean = 127.5) # swapRB

    # feed input to NN and attain its output
    network.setInput(blob)
    detections = network.forward()

    # defining storing objects

    raw_ROI_coords: Dict[int, Tuple[float,float,float]] = dict() # stores raw 3d coords (x,y,z) of center of ROI bbox

    pixel_ROI_coords: Dict[int, Tuple[int,int,int,int]] = dict() # stores pixel tl & br coords of bbox ROI

    pixel_center_ROI_coords: Dict[int, Tuple[float,float]] = dict() # store pixel mid-coordinates of bbox ROI

    close_ROI_dist: Dict[Tuple[int,int], float] = dict() # stores dist. between pair of ROIs that are too close 

    close_ROI_pairs: List[Tuple[int,int]] = [] # stores indices of ROI pairs that are too close 

    ROI_confidence: Dict[int, float] = dict() # store the confidence label of each ROI


    for i in range(detections.shape[2]):# looping total number of ROI objects predicted by model

        confidence = detections[0, 0, i, 2]


        # if our predictions meet a standard threshold confidence, continue
        if confidence > args["confidence"]:

            class_id = int(detections[0, 0, i, 1])

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype('int') # idx coordinates of bbox

            # Filtering only person objects detected in the frame. Class Id of 'person' is 15
            if class_id == 15.00:

                # store confidence of ROI object
                ROI_confidence[i] = confidence


                # store top-left and bottom-right coordinates of our ROI bbox (top-left = (startX, startY), bottom-right = (endX, endY))
                

                pixel_ROI_coords[i] = (startX, startY, endX, endY)

                # 
                find_raw_coords(i, (startX, startY), (endX, endY))

    # compute pair-wise distances between all combinations of ROI
    for i in raw_ROI_coords:
        for j in raw_ROI_coords:
            if i < j: # asserts that each comparison is a unique combination

                # stores the coordinates of all ROI pairs that are not maintaining social distancing
                are_ROIs_close(i,j)
                
    

    # holds all ROI indices
    ROI_indice = list(raw_ROI_coords.keys())

    # holds all ROI indices not maintaining social distancing (subset of ROI_indice)
    close_ROI_indice = np.unique(close_ROI_pairs)

    # holds all ROI indices maintaining social distancing (subset of ROI_indice)
    non_close_ROI = np.setdiff1d(ROI_indice, close_ROI_indice)


    # define annotation variables
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 2
    color = (0, 255, 0)

    # annotate ROIs that are maintaining social distancing
    for i in non_close_ROI:
        
        # unpack pixel bbox coords
        tl_x, tl_y, br_x, br_y = pixel_ROI_coords[i]

        # draw rectangle bbox
        cv2.rectangle(frame, (tl_x, tl_y), (br_x, br_y), color, thickness)
        
        # convert cms to ft
        depth_ft = cm_to_ft(raw_ROI_coords[i][2])

        # extract confidence of label
        conf = ROI_confidence[i]

        # annotate the depth ROI near its top-left bbox corner
        y = tl_y - 15 if tl_y - 15 > 15 else tl_y + 15
        txt = f'Depth: {depth_ft} ft. Conf: {conf:.3f}'
        txt_loc = (tl_x, y) 

        cv2.putText(frame, txt, txt_loc,
                    font, font_scale, color, thickness)

    # annotate pair of ROIs that are NOT maintaining social distancing
    color = (0, 0, 255)
    for pair in close_ROI_pairs:

        i,j = pair
        
        # unpack pixel bbox coords
        tl_x1, tl_y1, br_x1, br_y1 = pixel_ROI_coords[i]
        tl_x2, tl_y2, br_x2, br_y2 = pixel_ROI_coords[j]

        # draw rectangle bbox
        cv2.rectangle(frame, (tl_x1, tl_y1), (br_x1, br_y1), color, thickness)
        cv2.rectangle(frame, (tl_x2, tl_y2), (br_x2, br_y2), color, thickness)

        # draw a line from the center of both bbox
        x_mid1, y_mid1 = pixel_center_ROI_coords[i]
        x_mid2, y_mid2 = pixel_center_ROI_coords[j]

        # int() acts as a ceiling function for float numbers
        cv2.line(frame, (int(x_mid1) , int(y_mid1) ), (int(x_mid2) , int(y_mid2) ), color, thickness)

        # convert cms to ft
        depth_ft1 = cm_to_ft(raw_ROI_coords[i][2])
        depth_ft2 = cm_to_ft(raw_ROI_coords[j][2])
        dist_ft = cm_to_ft(close_ROI_dist[pair])

        # extract confidence of ROI labels
        conf1 = ROI_confidence[i]
        conf2 = ROI_confidence[j]

        # annotate the depth of each ROI near its top-left bbox corner

        # ROI i:
        y1 = tl_y1 - 15 if tl_y1 - 15 > 15 else tl_y1 + 15
        txt1 = f'Depth: {depth_ft1} ft. Conf: {conf1: .3f}'
        txt_loc1 = (tl_x1, y1) 

        cv2.putText(frame, txt1, txt_loc1,
                    font, font_scale, color, thickness)

        # ROI j:
        y2 = tl_y2 - 15 if tl_y2 - 15 > 15 else tl_y2 + 15
        txt2 = f'Depth: {depth_ft2} ft. Conf: {conf2: .3f}'
        txt_loc2 = (tl_x2, y2) # insert txt near top-left of ROI 

        cv2.putText(frame, txt2, txt_loc2, 
                font, font_scale, color, thickness)


        # annotate distance between pair of ROIs (insert at mid of distance
        x_line_mid = int(round((x_mid1 + x_mid2) / 2, 2))
        y_line_mid = int(round((y_mid1 + y_mid2) / 2, 2))
        line_loc = (x_line_mid, y_line_mid)
        
        line_txt = f'{dist_ft} ft.'

        cv2.putText(frame, line_txt,  line_loc,
                font, font_scale, color, thickness)



    cv2.namedWindow('Frame',cv2.WINDOW_NORMAL)

    if args['record']:
        out.write(frame)

    # Show frame
    cv2.imshow('Frame', frame)
    cv2.resizeWindow('Frame',800,600)

    key = cv2.waitKey(1) & 0xFF

    # Press `q` to exit
    if key == ord("q"):
        break

# Clean
cap.release()
out.release()
cv2.destroyAllWindows()



    








