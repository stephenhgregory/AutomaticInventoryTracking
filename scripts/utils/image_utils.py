import enum
import os
import re
import cv2
import numpy as np
from cv2 import dnn_superres

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging (1)
import tensorflow as tf
from object_detection.utils import visualization_utils as viz_utils
import matplotlib
import matplotlib.pyplot as plt
import warnings
from typing import List, Tuple
import pytesseract as tess
from utils import logging_utils

matplotlib.use('TkAgg')  # Set the backend of matplotlib to show plots in a separate window
warnings.filterwarnings('ignore')  # Suppress Matplotlib warnings
tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow logging (2)

# Enable dynamic memory allocation of the GPU(s)
gpus = tf.config.get_visible_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class SrModelType(enum.Enum):
    """
    SR model types supported by OpenCV-2
    """
    edsr = 1  # Best Performance, biggest size, slowest inference
    espcn = 2  # Small model with fast and accurate inference
    fsrcnn = 3  # Small model with even faster and accurate inference
    lapsrn = 4  # Medium sized model that can upscale by factor as high as 8


def get_sr_model_type(saved_sr_model: str) -> SrModelType:
    """
    Retrieves the SR Model Type from the pathname of the saved SR model (path to a .pb file)

    Parameters
    ----------
    saved_sr_model: path to the SR model

    Returns
    -------
    sr_model_type: An SrModelType representing the model type

    """
    if 'edsr' in saved_sr_model.lower():
        sr_model_type = SrModelType.edsr
    elif 'espcn' in saved_sr_model.lower():
        sr_model_type = SrModelType.espcn
    elif 'fsrcnn' in saved_sr_model.lower():
        sr_model_type = SrModelType.fsrcnn
    elif 'lapsrn' in saved_sr_model.lower():
        sr_model_type = SrModelType.lapsrn

    return sr_model_type


def get_sr_scale(saved_sr_model: str) -> int:
    """
    Retrieves the scale for SR model to upsample by (x2, x3, etc.)

    Parameters
    ----------
    saved_sr_model: path to the SR model

    Returns
    -------
    scale: The upsampling scale the model is trained for

    """
    root, _ = os.path.splitext(saved_sr_model)
    scale = int(root[len(root) - 1])
    return scale


@logging_utils.time_logger
def filter_text_bboxes(boxes: np.ndarray, probability_scores: np.ndarray = None, overlap_thresh=0.1,
                       max_num_boxes: int = 3):
    """
    Filters detected bounding boxes around regions of text

    Parameters
    ----------
    boxes: A uint16 numpy array of shape (n, 4) containing bounding box coordinates for all bounding boxes found
        Each element in boxes: x_min, y_min, x_max, y_max (normalized from 0 to 1)
    probability_scores: The probabilities associated with each bounding box
    overlap_thresh: The overlap threshold required to suppress boxes
    max_num_boxes: The maximum number of bounding boxes to return

    Returns
    -------
    The merged bounding boxes

    """
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes are integers, convert them to floats -- this
    # is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of filtered bounding boxes and list of already-filtered box indices
    filter_boxes = []
    selected_indices = []

    # Loop over all of the bounding boxes
    num_boxes = boxes.shape[0]
    for i in range(num_boxes):
        for j in range(num_boxes):
            if i == j or i in selected_indices or j in selected_indices:
                continue

            # Get the indices of the larger and smaller boxes
            larger_box_index = i if area(boxes[i]) > area(boxes[j]) else j
            smaller_box_index = i if area(boxes[i]) < area(boxes[j]) else j

            # If the IOU between two boxes is over the threshold...
            if intersection_over_union(boxes[i], boxes[j]) > overlap_thresh:

                # if the larger box is much larger than the smaller box...
                if area(boxes[larger_box_index]) > 4 * area(boxes[smaller_box_index]):
                    # Don't keep the smaller box by simply adding it to the "selected
                    # indices" list without actually selecting it
                    selected_indices.append(smaller_box_index)

                # Else, if they are within a certain proportion of eachother's size
                else:
                    # If the percent of vertical overlap of the smaller box
                    # with the larger box is greater than a percentage...
                    if get_percent_vertical_overlap(smaller_box=boxes[smaller_box_index],
                                                    larger_box=boxes[larger_box_index]) > 0.75:
                        # Add the indices to the selected list
                        selected_indices.append(smaller_box_index)
                        selected_indices.append(larger_box_index)

                        # Make a new bbox by merging the two and add to the filtered list of boxes
                        filter_boxes.append(merge_bboxes(boxes[smaller_box_index], boxes[larger_box_index]))

    # Add the boxes to filtered_boxes which have not been merged
    filter_boxes.extend([box for i, box in enumerate(boxes) if i not in set(selected_indices)])

    # Only take the top 2 largest boxes
    filter_boxes = sorted(filter_boxes, key=lambda x: area(x), reverse=True)[:2]

    return np.array(filter_boxes).astype('int')


@logging_utils.time_logger
def filter_text_bboxes_taking_top_probs(boxes: np.ndarray, probability_scores: np.ndarray = None, overlap_thresh=0.1,
                                        max_num_boxes: int = 3):
    """
    Filters detected bounding boxes around regions of text

    Parameters
    ----------
    boxes: A uint16 numpy array of shape (n, 4) containing bounding box coordinates for all bounding boxes found
        Each element in boxes: x_min, y_min, x_max, y_max (normalized from 0 to 1)
    probability_scores: The probabilities associated with each bounding box
    overlap_thresh: The overlap threshold required to suppress boxes
    max_num_boxes: The maximum number of bounding boxes to return

    Returns
    -------
    The merged bounding boxes

    """
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes are integers, convert them to floats -- this
    # is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of filtered bounding boxes and list of already-filtered box indices
    filter_boxes = []
    selected_indices = []

    # Initialize list of probabilities
    filtered_probs = []

    # Loop over all of the bounding boxes
    num_boxes = boxes.shape[0]
    for i in range(num_boxes):
        for j in range(num_boxes):
            if i == j or i in selected_indices or j in selected_indices:
                continue

            # Get the indices of the larger and smaller boxes
            larger_box_index = i if area(boxes[i]) > area(boxes[j]) else j
            smaller_box_index = i if area(boxes[i]) < area(boxes[j]) else j

            # If the IOU between two boxes is over the threshold...
            if intersection_over_union(boxes[i], boxes[j]) > overlap_thresh:

                # if the larger box is much larger than the smaller box...
                if area(boxes[larger_box_index]) > 4 * area(boxes[smaller_box_index]):
                    # Don't keep the smaller box by simply adding it to the "selected
                    # indices" list without actually selecting it
                    selected_indices.append(smaller_box_index)

                # Else, if they are within a certain proportion of eachother's size
                else:
                    # If the percent of vertical overlap of the smaller box
                    # with the larger box is greater than a percentage...
                    if get_percent_vertical_overlap(smaller_box=boxes[smaller_box_index],
                                                    larger_box=boxes[larger_box_index]) > 0.75:
                        # Add the indices to the selected list
                        selected_indices.append(smaller_box_index)
                        selected_indices.append(larger_box_index)

                        # Make a new bbox by merging the two and add to the filtered list of boxes
                        filter_boxes.append(merge_bboxes(boxes[smaller_box_index], boxes[larger_box_index]))

                        # Add the max probability from the merged box to the filtered list of probabilities
                        filtered_probs.append(
                            max(probability_scores[smaller_box_index], probability_scores[larger_box_index]))

    # Add the boxes to filtered_boxes which have not been merged
    filter_boxes.extend([box for i, box in enumerate(boxes) if i not in set(selected_indices)])

    # Add the probabilites from boxes that have not been filtered to filtered_probs
    filtered_probs.extend([prob for i, prob in enumerate(probability_scores) if i not in set(selected_indices)])

    # Get the boxes with the top "max_num_boxes" probabilities
    highest_probability_indices = np.argsort(filtered_probs)[-max_num_boxes:]
    highest_prob_boxes = [filter_boxes[i] for i in highest_probability_indices]

    return np.array(highest_prob_boxes).astype('int')


def get_percent_vertical_overlap(smaller_box, larger_box) -> np.ndarray:
    """
    Gets the percentage of smaller_box that is vertically overlapped by larger_box

    Parameters
    ----------
    smaller_box: The smaller bounding box
        Bounding box format: x_min, y_min, x_max, y_max (normalized from 0 to 1)
    larger_box: The larger bounding box
        Bounding box format: x_min, y_min, x_max, y_max (normalized from 0 to 1)

    Returns
    -------
    The percentage that the smaller box is vertical overlapped by the larger box
    """
    return (larger_box[3] - smaller_box[1]) / (smaller_box[3] - smaller_box[1]) if smaller_box[3] > larger_box[3] \
        else (larger_box[1] - smaller_box[3]) / (smaller_box[1] - smaller_box[3])


def area(box_a) -> int:
    """Returns the area of a bounding box with coordinates (x_min, y_min, x_max, y_max)"""
    return (box_a[3] - box_a[1]) * (box_a[2] - box_a[0])


def merge_bboxes(box_a, box_b) -> np.ndarray:
    """
    Merges two bounding boxes into one bounding box which encompasses both

    Parameters
    ----------
    box_a: The first bounding box
        Bounding box format: x_min, y_min, x_max, y_max (normalized from 0 to 1)
    box_b: The second bounding box
        Bounding box format: x_min, y_min, x_max, y_max (normalized from 0 to 1)

    Returns
    -------
    A uint8 numpy array representing the merged bounding box
    """
    start_x = min(box_a[0], box_b[0])
    start_y = min(box_a[1], box_b[1])
    end_x = max(box_a[2], box_b[2])
    end_y = max(box_a[3], box_b[3])

    return np.array([start_x, start_y, end_x, end_y])


def intersection_over_union(box_a, box_b) -> float:
    """
    Finds the Intersection-Over-Union (IOU) score between two bounding boxes

    Parameters
    ----------
    box_a: The first bounding box
    box_b: The second bounding box

    Returns
    -------
    The IOU between both bounding boxes, a percentage/ratio (between 0 and 1)
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    # compute the area of intersection rectangle
    area_of_intersection = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = area_of_intersection / float(box_a_area + box_b_area - area_of_intersection)

    # return the intersection over union value
    return iou


def non_max_suppression(boxes: np.ndarray, probability_scores=None, overlap_thresh: float = 0.3,
                        bbox_type: str = 'label'):
    """
    Run non-max suppression on bounding boxes to eliminate duplicate bounding boxes

    Parameters
    ----------
    boxes: The bounding box coordinates for all bounding boxes found
        Each element in boxes: y_min, x_min, y_max, x_max (normalized from 0 to 1)
    probability_scores: The probabilities (confidences) associated with each bounding box
    overlap_thresh: The overlap threshold required to suppress boxes
    bbox_type: Either 'label' or 'textbox', the type of bounding box annotations.
        'label' type boxes: y_min, x_min, y_max, x_max (normalized from 0 to 1)
        'textbox' type boxes: x_min, y_min, x_max, y_max (normalized from 0 to 1)

    Returns
    -------
    Only the boxes that were picked AND the probability scores for those boxes

    """
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes are integers, convert them to floats -- this
    # is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    if bbox_type == 'label':
        y1 = boxes[:, 0]
        x1 = boxes[:, 1]
        y2 = boxes[:, 2]
        x2 = boxes[:, 3]
    elif bbox_type == 'textbox':
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

    # compute the area of the bounding boxes and grab the indexes to sort
    # (in the case that no probabilities are provided, simply sort on the
    # bottom-left y-coordinate)
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    indexes = y2

    # if probabilities are provided, sort on them instead
    if probability_scores is not None:
        indexes = probability_scores

    # sort the indexes
    indexes = np.argsort(indexes)

    # keep looping while some indexes still remain in the indexes list
    while len(indexes) > 0:
        # grab the last index in the indexes list and add the index value
        # to the list of picked indexes
        last = len(indexes) - 1
        i = indexes[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of the bounding
        # box and the smallest (x, y) coordinates for the end of the bounding
        # box
        xx1 = np.maximum(x1[i], x1[indexes[:last]])
        yy1 = np.maximum(y1[i], y1[indexes[:last]])
        xx2 = np.minimum(x2[i], x2[indexes[:last]])
        yy2 = np.minimum(y2[i], y2[indexes[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[indexes[:last]]

        # delete all indexes from the index list that have overlap greater
        # than the provided overlap threshold
        indexes = np.delete(indexes, np.concatenate(([last],
                                                     np.where(overlap > overlap_thresh)[0])))

    # return only the bounding boxes that were picked and the associated confidence scores
    return boxes[pick].astype("int"), np.array(probability_scores)[pick]


def EAST_old(image: np.ndarray, east_model: any, resize_width: int = 320, resize_height: int = 320,
             min_confidence: float = 0.5) -> None:
    """
    This function takes in an image, it runs the EAST model on the image, and extracts text from it,
    but then skips the displaying image part and just displays the text found in the image...
    Will have to be adjusted to return the correct text from everything it finds

    Parameters
    ----------
    image: uint8 numpy array representing an input image
    east_model: The EAST Detection model
    resize_width: The width (in px) to resize image to for text bounding box detection
    resize_height: The height (in px) to resize image to for text bounding box detection
    min_confidence: The minimum required confidence to keep a bounding box

    Returns
    -------
    None
    """

    # adding a 10% white border to the image
    top = int(0.1 * image.shape[0])
    bottom = top
    left = int(0.1 * image.shape[1])
    right = left
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    # Saving the original image and shape with a white border to put bounding boxes on later
    orig = image.copy()
    (origH, origW) = image.shape[:2]

    # Calculate the ratio between original and new image for both height and weight.
    # This ratio will be used to translate bounding box location on the original image.
    rW = origW / float(resize_width)
    rH = origH / float(resize_height)

    # resize the original image to new dimensions
    image = cv2.resize(image, (resize_width, resize_height))
    (H, W) = image.shape[:2]

    # construct a blob from the image to forward pass it to EAST model
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)

    # The following two layer need to pulled from EAST model for achieving this.
    layer_names = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]

    # Forward pass the blob from the image to get the desired output layers
    east_model.setInput(blob)
    (scores, geometry) = east_model.forward(layer_names)

    # Find predictions and apply non-maxima suppression which is an algo
    # to select best bounding box of any overlapping boxes (basically assuming they are the same object)
    (boxes, confidence_val) = process_east_predictions(scores, geometry, min_confidence)
    boxes = non_max_suppression(np.array(boxes), probability_scores=confidence_val)

    # Done with EAST now, moving on to...
    #
    ## Text Detection and Recognition

    # initialize the list of results
    results = []
    # loop over the bounding boxes to find the coordinate of bounding boxes
    for (startX, startY, endX, endY) in boxes:
        # scale the coordinates based on the respective ratios in order to
        # reflect bounding box on the original image
        # but also adding shift to try and include the whole text strip
        # since bounding box is often slightly too small for tesseract
        startX = int(startX * rW) - 5
        startY = int(startY * rH) - 5
        endX = int(endX * rW) + 5
        endY = int(endY * rH) + 5

        # extract the region of interest
        r = orig[startY:endY, startX:endX]

        # adding white border to each bounding box for better extraction of text
        top = int(0.1 * r.shape[0])
        bottom = top
        left = int(0.1 * r.shape[1])
        right = left
        # this function sometimes returns a NoneType object
        # (seemingly when the shift caused the start/endpoints to go off the page)
        r = cv2.copyMakeBorder(r, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255, 255, 255])

        # configuration settings for pytesseract to convert image to string.
        configuration = "-l eng --oem 1 --psm 7"
        # if statement to protect from errors caused by NoneType objects sent to tesseract
        if not (r is None):
            # This will recognize the text from the image of bounding box
            text = tess.image_to_string(r, config=configuration)
            # append box coordinate and associated text to the list of results
            results.append(((startX, startY, endX, endY), text))
        else:
            # still add box coordinates, but label that it caused an error
            results.append(((startX, startY, endX, endY), 'error'))


@logging_utils.time_logger
def detect_text(image: np.ndarray, text_bboxes: List[Tuple[int]], visualize: bool = False,
                image_name: str = None) -> List[str]:
    """
    Detects and returns text found in an input image

    Parameters
    ----------
    text_bboxes: The bounding box coordinates for each bounding box found in image
    image: uint8 numpy array representing an image with text (hopefully)
    visualize: True if we wish to visualize the resultant text and text boxes on the image
    image_name: TODO: Delete this parameter

    Returns
    -------
    detected_text: A list of detected text from the image
    """
    # Initialize list of detections (coordinates along with detected text)
    detections = []

    # Apply a thin white border to the image (shows better text box detection performance)
    image = apply_white_border(image)

    # Loop over every detected bounding box
    for start_x, start_y, end_x, end_y in text_bboxes:
        # Extract the region of interest
        text_box_region = image[start_y:end_y, start_x:end_x]

        # Add a white border to the region for better text detection performance
        text_box_region = apply_white_border(text_box_region)

        # Call pytesseract to detect text in image
        text = str(tess.image_to_string(text_box_region, config="-l eng --oem 1 --psm 7"))

        # Filter the detected text
        text = filter_pytesseract_text(text)

        # append box coordinate and associated text to the list of results
        detections.append(((start_x, start_y, end_x, end_y), text))

    # Visualize the results if we wish to do so
    if visualize:
        visualize_text_predictions(image, detections)

    # Only return the detected text (not the bounding box coordinates)
    detected_text = [detection[1] for detection in detections]
    return detected_text


def visualize_text_predictions(image: np.ndarray, detections: List[Tuple[Tuple[int, int, int, int], str]],
                               image_name: str = None) -> None:
    """
    Visualizes text predictions from character recognition by drawing bounding boxes and
    text on the image and showing said image with OpenCV-2

    Parameters
    ----------
    image: uint8 numpy array representing image to be shown
    detections: A list of text detections, each of which containing bounding box coordinates
        and the detected text
    image_name: The name of the image to save. If not specified, no image will be saved.

    Returns
    -------
    None
    """
    # Loop through each detection, iteratively drawing detections on the image
    for (start_x, start_y, end_x, end_y), text in detections:
        # Display the detected text
        print(f'{text}\n')

        # Remove text with any unicode chars above 128
        text = "".join([x if ord(x) < 128 else "" for x in text]).strip()

        # Draw the bounding box on the image
        cv2.rectangle(image, (start_x, start_y), (end_x, end_y),
                      (0, 0, 255), 2)

        # Draw the start and end points of bounding box
        cv2.circle(image, (start_x, start_y), 2, (0, 255, 0), 3)
        cv2.circle(image, (end_x, end_y), 2, (255, 0, 0), 3)

        # Draw the detected text on the image near the bounding box
        cv2.putText(image, text, (start_x + 10, end_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (155, 0, 0), 2)

    if image_name is not None:
        cv2.imwrite(f'{image_name}_with_detections.png', image)

    # Display labeled image and wait for key press to move to next one
    show_image(image)


def filter_pytesseract_text(raw_text: str) -> str:
    """Filters the detected text from pytesseract"""
    # Remove excape sequences from text
    filtered_text = raw_text.strip()
    # Remove all non-alphanumeric characters from the text
    filtered_text = re.sub('[\W_]+', ' ', filtered_text, flags=re.UNICODE)
    return filtered_text


def visualize_text_predictions_no_text(image: np.ndarray,
                                       detections: List[Tuple[Tuple[int, int, int, int], str]],
                                       image_name: str = None) -> None:
    """
    Visualizes text predictions from character recognition by drawing bounding boxes and
    NOT DRAWING TEXT on the image.
    Shows said image with OpenCV-2.

    Parameters
    ----------
    image: uint8 numpy array representing image to be shown
    detections: A list of text detections, each of which containing bounding box coordinates
        and the detected text
    image_name: The name of the image

    Returns
    -------
    None
    """
    # Loop through each detection, iteratively drawing detections on the image
    for (start_x, start_y, end_x, end_y), text in detections:
        # Draw the bounding box on the image
        cv2.rectangle(image, (start_x, start_y), (end_x, end_y),
                      (0, 0, 255), 2)

        # Draw the start and end points of bounding box
        cv2.circle(image, (start_x, start_y), 2, (0, 255, 0), 3)
        cv2.circle(image, (end_x, end_y), 2, (255, 0, 0), 3)

    if image_name is not None:
        cv2.imwrite(f'{image_name}_with_detections.png', image)

    # Display labeled image and wait for key press to move to next one
    show_image(image)


def visualize_generic_bounding_boxes(image: np.ndarray,
                                     detections: List[Tuple[int, int, int, int]],
                                     image_name: str = None) -> None:
    """
    Visualizes generic bounding boxes of the form (start_x, start_y, end_x, end_y)
    by showing images with OpenCV-2

    Parameters
    ----------
    image: uint8 numpy array representing image to be shown
    detections: A list of text detections, each of which containing bounding box coordinates
    image_name: The name of the image

    Returns
    -------
    None
    """
    # Loop through each detection, iteratively drawing detections on the image
    for start_x, start_y, end_x, end_y in detections:
        # Draw the bounding box on the image
        cv2.rectangle(image, (start_x, start_y), (end_x, end_y),
                      (0, 0, 255), 2)

        # Draw the start and end points of bounding box
        cv2.circle(image, (start_x, start_y), 2, (0, 255, 0), 3)
        cv2.circle(image, (end_x, end_y), 2, (255, 0, 0), 3)

    if image_name is not None:
        cv2.imwrite(f'{image_name}_with_detections.png', image)

    # Display labeled image and wait for key press to move to next one
    show_images(np.array([image]), [image_name] or ["image"])


def process_east_predictions(prob_score, geo, min_confidence: float = 0.5):
    """
    Processes predictions from EAST neural network.
    Returns a bounding box and probability score if the prediction is more than
    the minimum confidence.

    Parameters
    ----------
    prob_score: TODO: Documentation
    geo: TODO: Documentation
    min_confidence: TODO: Documentation

    Returns
    -------
    TODO: Documentation

    """

    (numR, numC) = prob_score.shape[2:4]
    boxes = []
    confidence_val = []

    # loop over rows
    for y in range(0, numR):
        # TODO: Comments
        scoresData = prob_score[0, 0, y]
        x0 = geo[0, 0, y]
        x1 = geo[0, 1, y]
        x2 = geo[0, 2, y]
        x3 = geo[0, 3, y]
        anglesData = geo[0, 4, y]

        # loop over the number of columns
        for i in range(0, numC):
            if scoresData[i] < min_confidence:
                continue

            # TODO: Comments
            (offX, offY) = (i * 4.0, y * 4.0)

            # extracting the rotation angle for the prediction and computing the sine and cosine
            angle = anglesData[i]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # using the geo volume to get the dimensions of the bounding box
            height = x0[i] + x2[i]
            width = x1[i] + x3[i]

            # compute start and end for the text pred bbox
            end_x = int(offX + (cos * x1[i]) + (sin * x2[i]))
            end_y = int(offY - (sin * x1[i]) + (cos * x2[i]))
            start_x = int(end_x - width)
            start_y = int(end_y - height)

            boxes.append((start_x, start_y, end_x, end_y))
            confidence_val.append(scoresData[i])

    # return bounding boxes and associated confidence_val
    return boxes, confidence_val


def apply_white_border(image: np.ndarray, size: float = 0.1) -> np.ndarray:
    """
    Applies a white border to an input image, with a size determined by the size parameter

    Parameters
    ----------
    image: uint8 numpy array representing the input image
    size: The size of the white border to be placed on the image

    Returns
    -------
    A new numpy array representing the image with a white border
    """
    # adding a (size)% white border to the image
    top = int(size * image.shape[0]) if int(size * image.shape[0]) > 0 else 1
    bottom = top
    left = int(size * image.shape[1])
    right = left
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    assert image is not None
    return image


@logging_utils.time_logger
def detect_text_boxes(image: np.ndarray, east_model: any, new_width: int = 320, new_height: int = 320,
                      min_score_thresh: float = 0.5, padding: int = 10) -> np.ndarray:
    """
    Runs EAST (An Efficient and Accurate Scene Text Detector) on an input image to detect text boxes.

    Parameters
    ----------
    image: uint8 numpy array representing an input image
    east_model: The EAST Detection model
    new_width: The target width used to resize the image
    new_height: The target height used to resize the image
    min_score_thresh: The minimum score required to keep a text box
    padding: Number of pixels with which to pad (expand) each text bounding box.
        Facilitates easier text detection by PyTesseract.

    Returns
    -------
    uint8 numpy array representing detected boxes
    """
    # Make a copy of the image so that resizing doesn't affect the original
    image = image.copy()

    # adding a 10% white border to the image
    image = apply_white_border(image, size=0.1)

    # Get the resizing ratios for translating bounding boxes later
    original_height, original_width = image.shape[:2]
    width_ratio = original_width / new_width
    height_ratio = original_height / new_height

    # resize the original image to new dimensions
    image = cv2.resize(image, (new_width, new_height))

    # construct a blob from the image to forward pass it to EAST model
    blob = cv2.dnn.blobFromImage(image, 1.0, (new_width, new_height),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)

    # The following two layer need to pulled from EAST model for achieving this.
    layer_names = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]

    # Forward pass the blob from the image to get the desired output layers
    east_model.setInput(blob)
    scores, geometry = east_model.forward(layer_names)

    # Find predictions and apply NMS (non-maximum suppression) and our own filtering function
    text_boxes, confidence_val = process_east_predictions(scores, geometry, min_confidence=min_score_thresh)

    # Exit the function
    if len(text_boxes) == 0:
        return None

    # Apply NMS (non-maximum suppression) and proprietary filtering function
    text_boxes, confidence_val = non_max_suppression(np.array(text_boxes), probability_scores=confidence_val)

    # Realign the bounding boxes for original, un-resized image
    translated_text_boxes = []
    for start_x, start_y, end_x, end_y in text_boxes:
        # Pad the bounding box for easier detection
        start_x = int(start_x * width_ratio) - padding if int(start_x * width_ratio) >= padding else 0
        start_y = int(start_y * height_ratio) - padding if int(start_y * height_ratio) >= padding else 0
        end_x = int(end_x * width_ratio) + padding if int(end_x * width_ratio) + padding < original_width else int(
            end_x * width_ratio)
        end_y = int(end_y * height_ratio) + padding if int(end_y * height_ratio) + padding < original_height else int(
            end_y * height_ratio)
        translated_text_boxes.append([start_x, start_y, end_x, end_y])

    # Run further filtering of text boxes
    translated_text_boxes = filter_text_bboxes(np.array(translated_text_boxes), confidence_val, overlap_thresh=0.01,
                                               max_num_boxes=4)

    return translated_text_boxes


def resize_with_aspect_ratio(image: np.ndarray, width: int = None, height: int = None, inter=cv2.INTER_AREA):
    """
    Resizes an image by a given height OR a given width while retaining aspect ratio. (Only need to pass in one)
    NOTE: if you provide width AND height, height will be discarded, and width will be used to preserve aspect ratio.

    Parameters
    ----------
    image: A uint8 numpy array representing the image to be resized
    width: The desired width in pixels of the resized image
    height: The desired height in pixels of the resized image
    inter: The desired pixel interpolation method for image resizing

    Returns
    -------
    A uint8 numpy array representing the resized image

    :return:
    """
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


# TODO: Delete this function... there's little use for it.
def log_barcodes_with_visualizations(image: np.ndarray, barcodes: any, image_name: str) -> None:
    """
    Logs barcode detections to a logger file and visualizes barcode detections with
    boxes and barcode detections drawn on input

    Parameters
    ----------
    image: numpy array with shape (img_height, img_width, 3)
    barcodes: contains the detected barcodes from "image"
    image_name: The name with which to label the shown image

    Returns
    -------
    None

    """
    for barcode in barcodes:
        # extract the bounding box location of the barcode and draw the
        # bounding box surrounding the barcode on the image
        (x, y, w, h) = barcode.rect
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # the barcode data is a bytes object so if we want to draw it on
        # our output image we need to convert it to a string first
        barcode_data = barcode.data.decode("utf-8")
        barcode_type = barcode.type

        # draw the barcode data and barcode type on the image
        text = f'{barcode_data} ({barcode_type})'
        cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 2)

        # print the barcode type and data to the terminal
        print(f'[INFO] Found {barcode_type} barcode: {barcode_data}')

    # get a resized version of the output image (just for logging)
    resized_image = resize_with_aspect_ratio(image, height=1080)

    # show the output image
    cv2.imshow(image_name, resized_image)
    cv2.waitKey(0)
    cv2.destroyWindow(image_name)


def superres_images(images: np.ndarray, model_path: str, model_type: SrModelType, scale: int = 2,
                    visualize: bool = False) -> np.ndarray:
    """
    Upscales a numpy array of images with a super-resolution neural network.

    Parameters
    ----------
    images: uint8 numpy array with shape (num_images, img_height, img_width, 3)
    model_path: path to Super-Resolution model file (<model_name>.pb)
    model_type: The type of SR model used ("edsr", "espcn", "fsrcnn", "LapSRN")
    scale: The scale with which to upscale the image
    visualize: True if we wish to visualize images before and after Super-Resolution

    Returns
    -------
    upscaled_images: uint8 numpy array with shape (num_images, img_height * scale, img_width * scale, 3)

    """
    for i in range(images.shape[0]):
        images[i] = superres_image(images[i], model_path, model_type, scale, visualize=visualize)

    return images


@logging_utils.time_logger
def superres_image(image: np.ndarray, model_path: str, model_type: SrModelType, scale: int = 2,
                   visualize: bool = False) -> np.ndarray:
    """
    Upscales an image with a super-resolution neural network

    Parameters
    ----------
    image: uint8 numpy array with shape (img_height, img_width, 3)
    model_path: path to Super-Resolution model file (<model_name>.pb)
    model_type: The type of SR model used ("edsr", "espcn", "fsrcnn", "LapSRN")
    scale: The scale with which to upscale the image
    visualize: True if we wish to visualize the effects of Super-Resolution

    Returns
    -------
    upscaled_image: uint8 numpy array with shape (img_height * scale, img_width * scale, 3)

    """
    # Create a super-resolution (SR) object
    sr = dnn_superres.DnnSuperResImpl_create()
    # Read the desired model
    sr.readModel(model_path)
    # Set the desired model and scale to get correct pre- and post-processing
    sr.setModel(model_type.name, scale)
    # Upscale the image
    upscaled_image = sr.upsample(image)

    if visualize:
        crops = np.array([image, upscaled_image])
        crop_names = ['Before SR', 'After SR']
        show_images(images=crops, names=crop_names)

    return upscaled_image


def show_bounding_boxes(original_image: np.ndarray, detection_boxes: np.ndarray) -> None:
    """
    Show all of the predicted bounding boxes in an image with plt.show()
    Zooms in on each bounding box and shows it with plt.show()

    Parameters
    ----------
    original_image: uint8 numpy array with shape (img_height, img_width, 3)
    detection_boxes: The bounding boxes representing the regions of interest

    Returns
    -------
    None

    """
    for detection_box in detection_boxes:
        crop = crop_from_bounding_box(original_image, detection_box)
        plt.figure()
        plt.imshow(crop)
        plt.show()


def show_images(images: np.ndarray, names: List[str]) -> None:
    """
    Shows a series of numpy array with pyplot.show()

    Parameters
    ----------
    images: numpy array with shape (num_images, height, width, 3)
    names: The names to use as the titles for each of the images

    Returns
    -------
    None

    """
    num_rows = 1
    num_cols = images.shape[0]

    fig = plt.figure()

    for index in range(1, num_cols + 1):
        ax = fig.add_subplot(num_rows, num_cols, index)
        imgplot = plt.imshow(images[index - 1])
        ax.set_title(names[index - 1])

    plt.show()


def show_image(image: np.ndarray) -> None:
    """
    Shows a numpy array with pyplot.show()

    Parameters
    ----------
    image: numpy array with shape (height, width, 3)

    Returns
    -------
    None

    """
    plt.figure()
    plt.imshow(image)
    plt.show()


def get_bounding_box_crops(original_image: np.ndarray, detection_boxes: np.ndarray) -> np.ndarray:
    """
    Uses bounding boxes to return an np.ndarray of cropped sub-images, where each image is enclosed by a bbox

    Parameters
    ----------
    original_image: uint8 numpy array with shape (img_height, img_width, 3)
    detection_boxes: The bounding boxes representing the regions of interest

    Returns
    -------
    image_crops: uint8 numpy array with shape (detection_boxes.shape[0], img_height, img_width, 3)

    """
    image_crops = []
    for detection_box in detection_boxes:
        image_crops.append(crop_from_bounding_box(original_image, detection_box))
    return np.array(image_crops)


def crop_from_bounding_box(original_image: np.ndarray, detection_box: np.ndarray) -> np.ndarray:
    """
    Zooms in on a bounding box and returns that bounding box

    Parameters
    ----------
    original_image: uint8 numpy array with shape (img_height, img_width, 3)
    detection_box: The bounding box representing the regions of interest.
        Each element in detection_box: y_min, x_min, y_max, x_max (normalized from 0 to 1)

    Returns
    -------
    zoomed_in_image: uint8 numpy array with shape((denormed)y_max-(denormed)y_min, (denormed)x_max-(denormed)x_min, 3)

    """
    # Get and denormalize components of bounding box
    y_min, x_min, y_max, x_max = detection_box
    y_min = int(y_min * original_image.shape[0])
    x_min = int(x_min * original_image.shape[1])
    y_max = int(y_max * original_image.shape[0])
    x_max = int(x_max * original_image.shape[1])

    # Create and show the zoomed-in image
    zoomed_in_image = np.copy(original_image)
    zoomed_in_image = zoomed_in_image[y_min:y_max, x_min:x_max, :]

    return zoomed_in_image


def load_image_into_numpy_array(path: str) -> np.array:
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Parameters
    ----------
    path: the file path to the image

    Returns
    -------
    uint8 numpy array with shape (img_height, img_width, 3)
    """
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return np.array(image)


@logging_utils.time_logger
def infer_bounding_boxes(image_np: np.ndarray, image_name: str, detect_fn: any, category_index: list,
                         min_score_thresh: float = 0.3, visualize: bool = False) -> np.array(any):
    """
    Runs inference on an image given a path to that image

    Parameters
    ----------
    image_np: A uint8 numpy array representing the input image
    image_name: The name of the input image
    detect_fn: The object detection model used for inference
    category_index: Categories list derived from label map file
    min_score_thresh: The minimum confidence score required to keep each box
    visualize: True if we wish to visualize the image with bounding boxes

    Returns
    -------
    Detections from the object detection model
    """
    # print(f'Running inference for {image_name}')

    # Convert image to grayscale
    # image_np = np.tile(
    #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # input_tensor = np.expand_dims(image_np, 0)
    detections = detect_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    # Filter out the detections with a score which is too low
    detections = filter_detections_by_score(detections, min_score_thresh)

    if visualize:
        # Create a copy of the image
        image_np_with_detections = image_np.copy()
        # Draw bounding boxes, labels, and scores on the image
        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=min_score_thresh,
            agnostic_mode=False,
            line_thickness=10)
        # Show the images
        show_images(np.array([image_np_with_detections, image_np]), [f'{image_name} with detections', image_name])

        ''' Just logging (saving image_np_with_detections and original image) '''
        image_np_with_detections = cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f'{image_name}_with_detections.png', image_np_with_detections)
        cv2.imwrite(f'{image_name}.png', image_np)

    return detections


def filter_detections_by_score(detections: any, min_score_thresh: float = 0.3) -> any:
    """
    Filters Detections by a minimum score threshold

    Parameters
    ----------
    detections: The object detections, including bboxes, classes, and scores
    min_score_thresh: The minimum score threshold required to retain a given box

    Returns
    -------
    filtered_detections: The filtered object detections

    """
    boxes = []
    classes = []
    scores = []
    for i in range(detections['detection_boxes'].shape[0]):
        if detections['detection_scores'][i] > min_score_thresh:
            boxes.append(detections['detection_boxes'][i])
            classes.append(detections['detection_classes'][i])
            scores.append(detections['detection_scores'][i])
    filtered_detections = {
        'detection_boxes': np.array(boxes),
        'detection_classes': np.array(classes),
        'detection_scores': np.array(scores)
    }
    return filtered_detections
