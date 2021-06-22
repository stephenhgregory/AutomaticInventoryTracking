from pyzbar import pyzbar
import argparse
import cv2
from utils import image_utils, logger
import os
import pytesseract as tess

# Parse command-line arguments
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image_dir', required=True, help='Path to directory of input images')
args = vars(ap.parse_args())


def main():

    # Get the directory of images
    image_dir = args['image_dir']

    # Iterate over each image
    for image_name in os.listdir(args['image_dir']):
        if image_name.endswith('.jpg') or image_name.endswith('.png') \
                or image_name.endswith('.JPG') or image_name.endswith('.PNG'):

            # Load the input image
            image = cv2.imread(os.path.join(image_dir, image_name))
            
            # Preliminary running of pytesseract on total extracted label
            text = tess.image_to_string(image)
            print(text)

            # Find the barcodes in the image and deduce each of said barcode
            barcodes = pyzbar.decode(image)
            logger.log_num_barcodes(len(barcodes), image_name)

            # loop over the detected barcodes
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
            resized_image = image_utils.resize_with_aspect_ratio(image, height=1080)

            # show the output image
            cv2.imshow(image_name, resized_image)
            cv2.waitKey(0)
            cv2.destroyWindow(image_name)


if __name__ == '__main__':
    main()
