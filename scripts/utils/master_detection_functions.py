"""Contains master functions for detecting text/barcode/label information from images"""

import os
from typing import List, Tuple
import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging (1)
import tensorflow as tf
from object_detection.utils import label_map_util
import cv2
from utils import image_utils, image_augmentation_utils, logging_utils


@logging_utils.time_logger
def read_text_from_images(saved_label_detection_model: str, saved_textbox_detection_model: str,
                          saved_SR_model: str, image_dir: str, label_map_path: str,
                          superres: bool, preprocess_label_crops: bool,
                          visualize_label_detection: bool, visualize_superres: bool, visualize_textbox_detection: bool,
                          visualize_text_detection, min_label_score_thresh: float,
                          min_textbox_score_thresh: float) -> List[Tuple[List[str], str]]:
    """
    Reads information from images in a particular Directory

    Parameters
    ----------
    saved_label_detection_model: Path to saved label detection model
    saved_textbox_detection_model: Path to saved textbox detection model
    saved_SR_model: Path to saved super-resolution model
    image_dir: Path to folder of images to detect text from
    label_map_path: Path to label map (for label object detection)
    superres: Whether or not to perform Super-Resolution of detected
        text boxes
    preprocess_label_crops: Whether or not to apply preprocessing to label crops
    visualize_label_detection: Whether or not to visualize label detection
    visualize_superres: Whether or not to visualize the effects of super-resolution
    visualize_textbox_detection: Whether or not to visualize text box detection
    visualize_text_detection: Whether or not to visualize text detection
    min_label_score_thresh: The minimum confidence required to keep a detected
        label bounding box
    min_textbox_score_thresh: The minimum confidence required to keep a detected
        text bounding box


    Returns
    -------
    List of detected strings of text

    """
    # Initialize list of detected strings
    detected_strings = []

    if superres:
        # Get the Super-Resolution model type and scale
        sr_model_type = image_utils.get_sr_model_type(saved_SR_model)
        sr_scale = image_utils.get_sr_scale(saved_SR_model)

    # Load saved label detection model and textbox detection model
    bbox_detector = tf.saved_model.load(saved_label_detection_model)
    east_model = cv2.dnn.readNet(saved_textbox_detection_model)
    # Load the label map
    category_index = label_map_util.create_category_index_from_labelmap(label_map_path, use_display_name=True)

    # Get list of image file names
    file_names = [file_name for file_name in os.listdir(image_dir) if os.path.join(image_dir, file_name).endswith('.jpg') or os.path.join(image_dir, file_name).endswith('.JPG')]

    # Iterate over the images
    for file_name in tqdm.tqdm(file_names, desc="All Files"):

        # TODO: DELETE THIS #######
        if '1066' not in file_name:
            continue
        ###########################

        # Get the filename without the extension for saving images
        file_name_no_extension, _ = os.path.splitext(file_name)

        # Load the input image
        original_image = image_utils.load_image_into_numpy_array(os.path.join(image_dir, file_name))

        # Get the bounding box predictions for the labels
        detections = image_utils.infer_bounding_boxes(original_image, file_name_no_extension,
                                                      bbox_detector, category_index,
                                                      min_label_score_thresh, visualize_label_detection)

        # detection_boxes is np.array(shape=(n,4)) (Values scaled from 0 to 1)
        # Each box in detection_boxes: ymin, xmin, ymax, xmax
        detection_boxes = detections['detection_boxes']
        bbox_crops = image_utils.get_bounding_box_crops(original_image, detection_boxes)

        # Perform preprocessing on all labels
        if preprocess_label_crops:
            bbox_crops = image_augmentation_utils.preprocess_labels(bbox_crops)

        # Loop over all label crops...
        for label_crop_index, label_crop in tqdm.tqdm(enumerate(bbox_crops), desc=f"{file_name_no_extension} labels"):
            ''' Just logging
            cv2.imwrite(f'{file_name_no_extension}_crop{label_crop_index}.png', label_crop)
            '''

            # Perform super-resolution if we wish to do so
            if superres:
                print('doing super-resolution')  # TODO: DELETE THIS
                cv2.imwrite(f'{file_name_no_extension}_crop{label_crop_index}.png', label_crop)
                label_crop = image_utils.superres_image(label_crop, saved_SR_model, sr_model_type, sr_scale,
                                                        visualize=visualize_superres)
                cv2.imwrite(f'{file_name_no_extension}_crop{label_crop_index}_superres.png', label_crop)

            # Detect the text box coordinates
            text_bboxes = image_utils.detect_text_boxes(label_crop, east_model,
                                                        new_width=320, new_height=320,
                                                        min_score_thresh=min_textbox_score_thresh,
                                                        padding=10)

            # Simply continue to the next label if there are no text boxes found
            if text_bboxes is None:
                continue

            # Find the text in all of the boxes
            detected_text = image_utils.detect_text(label_crop, text_bboxes,
                                                    visualize=visualize_text_detection,
                                                    image_name=f'{file_name_no_extension}_crop{label_crop_index}')
            detected_strings.append((detected_text, file_name_no_extension))

    return detected_strings
