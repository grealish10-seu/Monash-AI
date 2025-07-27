import cv2
import tensorflow as tf
import numpy as np
import logging
import traceback

log = logging.getLogger(__name__)


def predict(model,src_img,dst_img):
    """Detects pose using Movenet and returns keypoints and annotated image."""
    interpreter = tf.lite.Interpreter(model_path=model)
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    img = cv2.imread(src_img)
    # Resize and normalize the image
    img_height, img_width, _ = img.shape
    input_shape = input_details[0]['shape'][1:3]  # Get expected input size
    resized_image = cv2.resize(img, input_shape)
    input_data = np.expand_dims(resized_image, axis=0).astype(np.float32) / 255.0

    # Perform inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Extract keypoints
    keypoints_output = interpreter.get_tensor(output_details[0]['index'])  # Shape: (1, 17, 3)
    # print(keypoints_output)
    keypoints = keypoints_output[0]  # Shape: (17, 3) - 17 keypoints, (y, x, confidence)
    # print(f"Keypoints: {keypoints[0][0]}")
    # Annotate the image
    annotated_image = img.copy()  # Important: draw on a copy

    # Connect the keypoints (example connections - customize as needed)
    connections = [
        (5, 6), (5, 11), (6, 12), (11, 12), (5, 7), (6, 8), (7, 9), (8, 10),
        (11, 13), (12, 14), (13, 15), (14, 16)
    ]

    for kp in keypoints[0]:
    #   print(f"{kp}")
        y, x, confidence = kp
    #   if confidence > 0.5:  # Adjust confidence threshold as needed
        x_coord = int(x * img_width)
        y_coord = int(y * img_height)
        cv2.circle(annotated_image, (x_coord, y_coord), 5, (0, 255, 0), -1) # Green circles


    for start, end in connections:
        y1, x1, c1 = keypoints[0][start]
        y2, x2, c2 = keypoints[0][end]
        # if c1 > 0.3 and c2 > 0.3: # enable confidence if necessary
        x1_coord = int(x1 * img_width)
        y1_coord = int(y1 * img_height)
        x2_coord = int(x2 * img_width)
        y2_coord = int(y2 * img_height)
        cv2.line(annotated_image, (x1_coord, y1_coord), (x2_coord, y2_coord), (0, 0, 255), 2)  # Red lines
    cv2.imwrite(dst_img, annotated_image)
    return keypoints


def main():
    try:
        log.info("Loading pose detection model...")
        model = 'movenet-full-256.tflite'
        log.info("Model is loaded.")
        # Predict on an image
        image_path = './test.jpg'  # Replace with your image path
        output_image = 'test_with_keypoints.jpg'
        result = predict(model,image_path,output_image)
        log.info(result)
    except Exception as e:
        print(f"Exception: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()