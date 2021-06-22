import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import pathlib
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import warnings
from absl import flags

matplotlib.use('TkAgg')                     # Set the backend of matplotlib to show plots in a separate window
warnings.filterwarnings('ignore')           # Suppress Matplotlib warnings
tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

# Enable dynamic memory allocation of the GPU(s)
gpus = tf.config.get_visible_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

flags.DEFINE_string('saved_model_dir', None, 'Path to saved model')
flags.DEFINE_string('image_dir', None, 'Path to directory of images to run inference on.')
flags.DEFINE_string('label_map', None, 'Path to label map.')
flags.DEFINE_bool('save_images', False, 'Whether or not to save images augmented with bounding box predictions.')
flags.DEFINE_string('save_image_dir', None, 'Path to directory of result predicted images.')

FLAGS = flags.FLAGS


def load_image_into_numpy_array(path: str) -> np.array:
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))


def run_inference(image_path: str, detect_fn: any, category_index: list, my_flags: any) -> None:
    """
    Runs inference on an image given a path to that image

    Args:
        image_path: (str) The path to a single image
        detect_fn: (TF Detection model) The object detection model used for inference
        category_index: (list) Categories list derived from label map file
        my_flags: (tf.app.flags) Command line flags

    Returns:
         None
    """
    print('Running inference for {}... '.format(image_path), end='')

    image_np = load_image_into_numpy_array(image_path)

    # Get the image name (without file extension)
    image_name = os.path.basename(image_path)
    image_name = os.path.splitext(image_name)[0]

    # Things to try:
    # Flip horizontally
    # image_np = np.fliplr(image_np).copy()

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

    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.30,
        agnostic_mode=False)

    plt.figure()
    plt.imshow(image_np_with_detections)
    plt.show()

    # If we wish to save the image with predicted bboxes and class labels drawn on...
    if my_flags.save_images:
        # If the result directory doesn't already exist, create it
        if not os.path.exists(my_flags.save_image_dir):
            os.mkdir(my_flags.save_image_dir)
        print(f'Saving image: {image_name}')
        plt.imsave(os.path.join(my_flags.save_image_dir, image_name + '.jpg'), image_np_with_detections)

    print('Done')


def main(unused_argv):
    flags.mark_flag_as_required('saved_model_dir')
    flags.mark_flag_as_required('image_dir')
    flags.mark_flag_as_required('label_map')
    flags.mark_flag_as_required('save_image_dir')
    tf.config.set_soft_device_placement(True)

    # Load saved model and build the detection function
    detect_fn = tf.saved_model.load(FLAGS.saved_model_dir)

    # Load the label map
    category_index = label_map_util.create_category_index_from_labelmap(FLAGS.label_map, use_display_name=True)

    # Iterate over the images
    for file_name in os.listdir(FLAGS.image_dir):
        if os.path.join(FLAGS.image_dir, file_name).endswith('.jpg') or \
                os.path.join(FLAGS.image_dir, file_name).endswith('.JPG'):
            run_inference(os.path.join(FLAGS.image_dir, file_name), detect_fn, category_index, FLAGS)


if __name__ == "__main__":
    tf.compat.v1.app.run()
