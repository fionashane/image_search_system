import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

# We are using an MobileNet Single Shot object detection model trained on the COCO dataset
# downloaded from Tensorflow Hub [model name: ssdlite_mobilenet_v2_coco_2018_05_09].
# This model can detect these object types in images:
ALL_LABELS = {
 1: 'person',
 2: 'bicycle',
 3: 'car',
 4: 'motorcycle',
 5: 'airplane',
 6: 'bus',
 7: 'train',
 8: 'truck',
 9: 'boat',
 10: 'traffic light',
 11: 'fire hydrant',
 13: 'stop sign',
 14: 'parking meter',
 15: 'bench',
 16: 'bird',
 17: 'cat',
 18: 'dog',
 19: 'horse',
 20: 'sheep',
 21: 'cow',
 22: 'elephant',
 23: 'bear',
 24: 'zebra',
 25: 'giraffe',
 27: 'backpack',
 28: 'umbrella',
 31: 'handbag',
 32: 'tie',
 33: 'suitcase',
 34: 'frisbee',
 35: 'skis',
 36: 'snowboard',
 37: 'sports ball',
 38: 'kite',
 39: 'baseball bat',
 40: 'baseball glove',
 41: 'skateboard',
 42: 'surfboard',
 43: 'tennis racket',
 44: 'bottle',
 46: 'wine glass',
 47: 'cup',
 48: 'fork',
 49: 'knife',
 50: 'spoon',
 51: 'bowl',
 52: 'banana',
 53: 'apple',
 54: 'sandwich',
 55: 'orange',
 56: 'broccoli',
 57: 'carrot',
 58: 'hot dog',
 59: 'pizza',
 60: 'donut',
 61: 'cake',
 62: 'chair',
 63: 'couch',
 64: 'potted plant',
 65: 'bed',
 67: 'dining table',
 70: 'toilet',
 72: 'tv',
 73: 'laptop',
 74: 'mouse',
 75: 'remote',
 76: 'keyboard',
 77: 'cell phone',
 78: 'microwave',
 79: 'oven',
 80: 'toaster',
 81: 'sink',
 82: 'refrigerator',
 84: 'book',
 85: 'clock',
 86: 'vase',
 87: 'scissors',
 88: 'teddy bear',
 89: 'hair drier',
 90: 'toothbrush'}

DETECTION_MODEL_DIR='detection_model'

def encode_labels(labels):
    """Returns a list of booleans (0/1) indicating the object types present in `labels`"""
    return [1 if label in labels else 0 for label in ALL_LABELS.values()]

def load_model():
    """Loads the detection model from directory DETECTION_MODEL_DIR"""
    model = tf.saved_model.load(DETECTION_MODEL_DIR)    
    return model.signatures['serving_default']

def detect_objects(image):
    """Detects objects in image and returns the set of labels for the objects"""
    model = load_model()

    image = np.asarray(image)
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]
    
    output_dict = model(input_tensor)

    num_detections = int(output_dict['num_detections'])
    detected_classes = output_dict['detection_classes'][0,:num_detections].numpy().astype(int)

    labels = set(ALL_LABELS[cls] for cls in detected_classes)
    return labels
