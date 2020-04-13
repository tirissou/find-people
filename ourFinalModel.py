#!/usr/bin/env python
import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
import PIL
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, scale_boxes
from yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body

#%matplotlib inline


def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = .6):

    box_scores = box_confidence * box_class_probs
 
    box_classes = K.argmax(box_scores, axis=-1)
    box_class_scores = K.max(box_scores, axis=-1)

 
    filtering_mask = box_class_scores >= threshold
  
    scores = tf.boolean_mask(box_class_scores, filtering_mask)
    boxes = tf.boolean_mask(boxes, filtering_mask)
    classes = tf.boolean_mask(box_classes, filtering_mask)
    
    return scores, boxes, classes

def iou(box1, box2):
    
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = (xi2 - xi1) * (yi2 - yi1)
 
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
   
    iou = inter_area / union_area
    
    return iou

def yolo_non_max_suppression(scores, boxes, classes, max_boxes = 10, iou_threshold = 0.5):
    
    
    max_boxes_tensor = K.variable(max_boxes, dtype='int32')     # tensor to be used in tf.image.non_max_suppression()
    K.get_session().run(tf.variables_initializer([max_boxes_tensor])) # initialize variable max_boxes_tensor
    
  
    nms_indices = tf.image.non_max_suppression( boxes, scores, max_boxes_tensor, iou_threshold)
    scores = tf.gather(scores, nms_indices)
    boxes = tf.gather(boxes, nms_indices)
    classes = tf.gather(classes, nms_indices)
    
    return scores, boxes, classes

def yolo_eval(yolo_outputs, image_shape = (720., 1280.), max_boxes=10, score_threshold=.6, iou_threshold=.5):
   
    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs
   
    # Convert boxes to be ready for filtering functions 
    boxes = yolo_boxes_to_corners(box_xy, box_wh)

    # Use one of the functions you've implemented to perform Score-filtering with a threshold of score_threshold (≈1 line)
    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = score_threshold)
    
    # Scale boxes back to original image shape.
    boxes = scale_boxes(boxes, image_shape)

    # Use one of the functions you've implemented to perform Non-max suppression with a threshold of iou_threshold (≈1 line)
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes = max_boxes, iou_threshold = iou_threshold)
    
   
    return scores, boxes, classes
pather = "/home/robot-tumas/Desktop/projects/Class/find-people/"

#Load pre_trained model
sess = K.get_session()
class_names = read_classes(pather + "classes.txt")
image_shape = (720., 1280.)    
yolo_model = load_model(pather +"compModel.pb")
#yolo_model.summary()
yolo_outputs = yolo_head(tf.convert_to_tensor(yolo_model.output), anchors, len(class_names))
input_image_shape = K.placeholder(shape=(2, ))
scores, boxes, classes = yolo_eval(yolo_outputs, input_image_shape)
import imageio

def predict(sess, image_file):
   

    # Preprocess your image
    image, image_data = preprocess_image(image_file, model_image_size = (608, 608))

    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes], feed_dict={yolo_model.input: image_data,
                                                                                       input_image_shape: [image.size[1], image.size[0]],
                                                                                       K.learning_phase(): 0})
 

    # Print predictions info
    Filter = []
    counter = 0 # counter of non humans
    for i in range(len(out_classes)):
        if out_classes[i] != 0:
            Filter.append(False)
            counter = counter+1
        else:
            Filter.append(True)
    numberValid = len(out_classes) - counter
    for i in range(len(out_classes)):
        out_classes[i] = Filter[i]*out_classes[i]
    out_classes = out_classes[0:numberValid]
    # Generate colors for drawing bounding boxes.
    colors = generate_colors(class_names)
    # Draw bounding boxes on the image file
    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    # Save the predicted bounding box on the image
    image.save(os.path.join("out", image_file), quality=90)
    # Display the results in the notebook
    output_image = scipy.misc.imread(os.path.join("out", image_file))
    imshow(output_image)
    imageio.imwrite("output.jpg",output_image)
    #print('output classes: ',out_classes.eval())
    return out_scores, out_boxes, out_classes
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', dest='imagePath',required=True, help='Required: location of image' )
    args = parser.parse_args()
    out_scores, out_boxes, out_classes = predict(sess, args.imagePath)
