import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 3
TEST_SIZE = 0.4


def convert_to_five_digit_str(number):
    string = str(number)

    if len(string) > 5:
        raise ValueError

    # Convert the string with the same value but a 5 letter string
    while len(string) != 5:
        string = "0" + string

    return string


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    # List of tuples, iamge and label
    data_dir_path = os.path.join(os.getcwd(), data_dir)
    image_labels = list()
    for img_dir in range(0, NUM_CATEGORIES):
        img_head_label_int = 0
        img_tail_label_int = 0
        # Go through all the images
        while True:
            img_head_label_str = convert_to_five_digit_str(img_head_label_int)
            img_tail_label_str = convert_to_five_digit_str(img_tail_label_int)

            image_path = data_dir_path + os.sep + str(img_dir) + os.sep + img_head_label_str + "_" + img_tail_label_str + ".ppm"
            image = cv2.imread(image_path)
            try:
                # Resize the image and add it to the list
                resized_image = cv2.resize(src=image, dsize=(IMG_HEIGHT, IMG_WIDTH))
                # Add the resized image to the list
                image_labels.append((resized_image, str(img_dir)))
                
                # Go to the next image
                img_tail_label_int += 1

            except (AttributeError, cv2.error):
                if img_tail_label_int == 0:
                    # End of all heads, so end of the images in the img_dir
                    # Go to next img_dir
                    break
                else:
                    # End of tail, go to next head in the same img_dir
                    img_head_label_int += 1
                    img_tail_label_int = 0

    # Seperating the images and labels
    images = [image_label[0] for image_label in image_labels]
    labels = [image_label[1] for image_label in image_labels]


    return (images, labels)


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """

    # Create a neural network
    model = tf.keras.models.Sequential()

    # Convolutional layer. Learn 32 filters using a 3x3 kernel
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))

    # Max pooling layer, using 2x2 pool size
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # Flatten units
    model.add(tf.keras.layers.Flatten())

    # Add a hidden layer, with ReLU activation
    model.add(tf.keras.layers.Dense(50, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.5))

    # Add output layer with NUM_CATEGORIES units, with softmax activation
    model.add(tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax"))

    # Compile the model
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    main()
