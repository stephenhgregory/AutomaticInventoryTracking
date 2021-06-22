"""Main function for scanning and detecting text from images in a given directory"""

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging (1)
import tensorflow as tf
import matplotlib
import warnings
from absl import flags
from utils import master_detection_functions
from mongo import mongo_utils

matplotlib.use('TkAgg')  # Set the backend of matplotlib to show plots in a separate window
warnings.filterwarnings('ignore')  # Suppress Matplotlib warnings
tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow logging (2)

# Enable dynamic memory allocation of the GPU(s)
# gpus = tf.config.get_visible_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)

# Enable specific memory limit of the GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [
            tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
    except RuntimeError as e:
        print(e)

flags.DEFINE_string('saved_object_detection_model',
                    '../tensorflow/object_detection_training/exported-models/my_ssd_mobilenet_v2_fpnlite/saved_model',
                    'Path to saved object detection model')
flags.DEFINE_string('saved_EAST_model',
                    '../tensorflow/EAST/frozen_east_text_detection.pb',
                    'Path to saved EAST model')
flags.DEFINE_string('saved_SR_model',
                    '../tensorflow/super_resolution/exported-models/FSRCNN_x2.pb',
                    'Path to saved super-resolution model')
flags.DEFINE_string('image_dir', '../images/Batch1', 'Path to directory of images to scan.')
flags.DEFINE_string('label_map', '../tensorflow/object_detection_training/annotations/label_map.pbtxt',
                    'Path to label map.')
flags.DEFINE_bool('save_results', False, 'Whether or not to save results to a database.')
flags.DEFINE_bool('superres', True, 'Whether or not to do Super-Resolution.')
flags.DEFINE_bool('preprocess_label_crops', False, 'Whether or not to do image preprocessing on the label crops.')
flags.DEFINE_bool('visualize_superres', False, 'Whether or not to visualize the results of Super-Resolution.')
flags.DEFINE_bool('visualize_label_detection', False, 'Whether or not to visualize the results of label detection.')
flags.DEFINE_bool('visualize_textbox_detection', False, 'Whether or not to visualize the results of text box detection.')
flags.DEFINE_bool('visualize_text_detection', False, 'Whether or not to visualize the results of text detection.')
flags.DEFINE_float('min_label_score_thresh', 0.5,
                   'Minimum object detection score required to use a bounding box around a label.')
flags.DEFINE_float('min_textbox_score_thresh', 0.3,
                   'Minimum object detection score required to use a bounding box around text inside of a label.')

FLAGS = flags.FLAGS


def main(unused_argv):
    flags.mark_flag_as_required('saved_object_detection_model')
    flags.mark_flag_as_required('saved_EAST_model')
    flags.mark_flag_as_required('saved_SR_model')
    flags.mark_flag_as_required('image_dir')
    flags.mark_flag_as_required('label_map')
    tf.config.set_soft_device_placement(True)

    # Detect text strings from batch of images
    # detected_strings is a List[Tuple[List[str], str]]
    detected_strings = master_detection_functions.read_text_from_images(
        saved_label_detection_model=FLAGS.saved_object_detection_model,
        saved_textbox_detection_model=FLAGS.saved_EAST_model,
        saved_SR_model=FLAGS.saved_SR_model,
        image_dir=FLAGS.image_dir,
        label_map_path=FLAGS.label_map,
        superres=FLAGS.superres,
        preprocess_label_crops=FLAGS.preprocess_label_crops,
        visualize_label_detection=FLAGS.visualize_label_detection,
        visualize_superres=FLAGS.visualize_superres,
        visualize_textbox_detection=FLAGS.visualize_textbox_detection,
        visualize_text_detection=FLAGS.visualize_text_detection,
        min_label_score_thresh=FLAGS.min_label_score_thresh,
        min_textbox_score_thresh=FLAGS.min_textbox_score_thresh)

    # Save the results
    if FLAGS.save_results:
        # Insert the detected text from this batch into the database
        mongo_utils.insert_batch_into_database(detected_strings, drop_collection=False)


if __name__ == "__main__":
    tf.compat.v1.app.run()
