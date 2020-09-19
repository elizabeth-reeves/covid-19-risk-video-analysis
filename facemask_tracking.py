# Libby Reeves
# 9/19/2020

import tensorflow as tf
from utils import visualization_utils as vis_util
import detect_mask_image
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import os
from utils import backbone


def cumulative_facemask_counting(input_video, detection_graph, category_index,
                                 is_color_recognition_enabled, roi, deviation,
                                 targeted_objects=None):
    """
    Count objects passing across the ROI; keep a cumulative count.

    """

    # input video
    cap = cv2.VideoCapture(input_video)

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    roi_position = int(roi * width)

    total_unmasked = 0
    total_masked = 0
    total_unknown = 0
    with detection_graph.as_default():
        with tf.compat.v1.Session(graph=detection_graph) as sess:

            print("[INFO] loading face detector model...")
            prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
            weightsPath = os.path.sep.join(["face_detector",
                                            "res10_300x300_ssd_iter_140000.caffemodel"])
            net = cv2.dnn.readNet(prototxtPath, weightsPath)

            # load the face mask detector model from disk
            print("[INFO] loading face mask detector model...")
            model = load_model("mask_detector.model")


            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # for all the frames that are extracted from input video
            while True:
                ret, frame = cap.read()

                if not ret:
                    print("end of the video file...")
                    break

                input_frame = frame

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(input_frame, axis=0)

                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                # insert information text to video frame
                font = cv2.FONT_HERSHEY_SIMPLEX


                (locs, preds) = detect_mask_image.detect_and_predict_mask(input_frame, net, model)


                # Visualization of the results of a detection.
                counter, csv_line, counting_mode = vis_util.visualize_boxes_and_labels_on_image_array_x_axis(
                    cap.get(1),
                    input_frame,
                    1,
                    is_color_recognition_enabled,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    targeted_objects=targeted_objects,
                    x_reference=roi_position,
                    deviation=deviation,
                    use_normalized_coordinates=True,
                    line_thickness=4)

                # detect_mask_image.mask_image_filename('face_mask_detection/images/pic1.jpeg')

                label = ''
                for (box, pred) in zip(locs, preds):
                    # unpack the bounding box and predictions
                    (startX, startY, endX, endY) = box
                    (mask, withoutMask) = pred

                    # determine the class label and color we'll use to draw
                    # the bounding box and text
                    label = "Mask" if mask > withoutMask else "No Mask"
                    color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

                    # include the probability in the label
                    label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
                    cv2.putText(frame, label, (startX, startY - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                    if "No" in label:
                        total_unmasked += counter
                    else:
                        total_masked += counter

                # when the vehicle passed over line and counted, make the color of ROI line green
                if counter == 1:
                    cv2.line(input_frame, (roi_position, 0),
                             (roi_position, height), (0, 0xFF, 0), 5)
                else:
                    cv2.line(input_frame, (roi_position, 0),
                             (roi_position, height), (0, 0, 0xFF), 5)

                if label == '':
                    total_unknown += counter

                # insert information text to video frame
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(
                    input_frame,
                    'Cumulative Detected Masked: ' + str(total_masked),
                    (10, 70),
                    font,
                    0.8,
                    (0, 0xFF, 0xFF),
                    2,
                    cv2.FONT_HERSHEY_SIMPLEX
                )
                cv2.putText(
                    input_frame,
                    'Cumulative Detected Unmasked: ' + str(total_unmasked),
                    (10, 35),
                    font,
                    0.8,
                    (0, 0xFF, 0xFF),
                    2,
                    cv2.FONT_HERSHEY_SIMPLEX
                )
                cv2.putText(
                    input_frame,
                    'Cumulative Unknown: ' + str(total_unknown),
                    (10, 105),
                    font,
                    0.8,
                    (0, 0xFF, 0xFF),
                    2,
                    cv2.FONT_HERSHEY_SIMPLEX
                )

                cv2.putText(
                    input_frame,
                    'ROI Line',
                    (545, roi_position - 10),
                    font,
                    0.6,
                    (0, 0, 0xFF),
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow('object counting', input_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video",
                    default=0,
                    help="path to input video")
    ap.add_argument("-r", "--roi",
                    default=0.5,
                    help="position of ROI")
    ap.add_argument("-d", "--deviation",
                    default=1,
                    help="margin from ROI for detection")
    args = vars(ap.parse_args())

    # By default I use an "SSD with Mobilenet" model here. See the detection model zoo (
    # https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc
    # /detection_model_zoo.md) for a list of other models that can be run out-of-the-box with
    # varying speeds and accuracies.
    detection_graph, category_index = backbone.set_model('ssd_mobilenet_v1_coco_2018_01_28',
                                                         'mscoco_label_map.pbtxt')

    is_color_recognition_enabled = 0

    cumulative_facemask_counting(args["video"], detection_graph, category_index,
                                 is_color_recognition_enabled, args["roi"], args["deviation"],
                                 targeted_objects="person")


if __name__ == "__main__":
    run()
