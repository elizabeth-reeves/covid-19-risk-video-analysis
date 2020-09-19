
import tensorflow as tf

# Object detection imports
from utils import backbone
from api import object_counting_api

# input_video = "./input_images_and_videos/traffic_light.mp4"
input_video = 0

# By default I use an "SSD with Mobilenet" model here. See the detection model zoo (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.
detection_graph, category_index = backbone.set_model('ssd_mobilenet_v1_coco_2018_01_28', 'mscoco_label_map.pbtxt')

#object_counting_api.object_counting(input_video, detection_graph, category_index, 0) # for counting all the objects, disabled color prediction

#object_counting_api.object_counting(input_video, detection_graph, category_index, 1) # for counting all the objects, enabled color prediction


targeted_objects = "person" # (for counting targeted objects) change it with your targeted objects
is_color_recognition_enabled = 0
roi = 500
deviation = 15

# object_counting_api.targeted_object_counting(input_video, detection_graph, category_index, is_color_recognition_enabled, targeted_objects) # targeted objects counting

object_counting_api.cumulative_object_counting_x_axis(input_video, detection_graph, category_index,
                                                      is_color_recognition_enabled, roi, deviation,
                                                      targeted_objects=targeted_objects)
