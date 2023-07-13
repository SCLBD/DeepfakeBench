import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import json
import logging
import os

def create_logger(log_path):
    # Create log path
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # Create logger object
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create file handler and set the formatter
    fh = logging.FileHandler(log_path)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(fh)

    # Add a stream handler to print to console
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)  # Set logging level for stream handler
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def calculate_average_frequency_spectra(images):
    if len(images) == 0:
        return None

    # Create an empty array to store the spectra
    avg_spectra = np.zeros(images[0].shape[:2], dtype=np.float32)  # Keep only the height and width information of the images

    # Iterate over the images
    for image in images:
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Perform high-pass filtering
        blurred_image = cv2.medianBlur(gray_image, 9)  # Increase the kernel size
        high_pass_image = gray_image - blurred_image

        # Compute the Fourier transform
        freq_spectrum = np.fft.fft2(high_pass_image)

        # Shift the zero-frequency component to the center
        freq_spectrum_shifted = np.fft.fftshift(freq_spectrum)

        # Accumulate the spectra
        avg_spectra += np.abs(freq_spectrum_shifted)

    # Calculate the average spectrum
    avg_spectra /= len(images)

    # Return the average spectrum
    return avg_spectra

def visualize_average_frequency_spectra(real_images, fake_images, filename):
    # Clear the plot canvas
    plt.clf()

    # Randomly select images from real and fake images
    selected_real_images = random.sample(real_images, k=min(len(real_images), 2000))
    selected_fake_images = random.sample(fake_images, k=min(len(fake_images), 2000))

    # Calculate the average frequency spectra for real images
    avg_spectra_real = calculate_average_frequency_spectra(selected_real_images)

    # Calculate the average frequency spectra for fake images
    avg_spectra_fake = calculate_average_frequency_spectra(selected_fake_images)

    if avg_spectra_real is None or avg_spectra_fake is None:
        print("No images to process.")
        return

    # Calculate the difference of the average frequency spectra
    diff_spectra = np.abs(avg_spectra_real - avg_spectra_fake)

    # Set up the plot
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    fig.suptitle(filename, fontsize=16)

    # Plot real images
    axes[0].imshow(np.log1p(avg_spectra_real), cmap='viridis')
    axes[0].set_title('Real Images')
    axes[0].axis('off')

    # Plot fake images
    axes[1].imshow(np.log1p(avg_spectra_fake), cmap='viridis')
    axes[1].set_title('Fake Images')
    axes[1].axis('off')

    # Plot the difference
    axes[2].imshow(diff_spectra, cmap='viridis')
    axes[2].set_title('Difference (Real - Fake)')
    axes[2].axis('off')

    # Save the plot
    path = f'img/{filename}_jet.png'
    plt.savefig(path)
    plt.close()

def load_images_from_paths(image_paths):
    images = []
    for image_path in image_paths:
        try:
            if not os.path.isfile(image_path):
                logger.error(f"Image file does not exist: {image_path}")
                continue

            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to load the image: {image_path}")
            else:
                # Convert the color image to the Lab color space
                lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
                # Split into L, a, b channels
                l, a, b = cv2.split(lab_image)
                # Apply histogram equalization to the L channel
                equalized_l = cv2.equalizeHist(l)
                # Merge the processed L channel with the original a, b channels
                equalized_lab = cv2.merge([equalized_l, a, b])
                # Convert the image back to the BGR color space
                equalized_image = cv2.cvtColor(equalized_lab, cv2.COLOR_Lab2BGR)
                images.append(equalized_image)
        except cv2.error as e:
            logger.error(f"Error loading image: {image_path}")
            logger.error(f"Error message: {str(e)}")
    return images

def load_images_from_json(json_data, real_images, fake_images):
    for value in json_data.values():
        if isinstance(value, dict):
            if "c23" in value:
                load_images_from_json(value["c23"], real_images, fake_images)
            else:
                load_images_from_json(value, real_images, fake_images)
        elif isinstance(value, list) and "frames" in json_data:
            frames = json_data["frames"]
            label = json_data["label"]
            images = load_images_from_paths(frames)
            if "real" in label.lower():
                real_images.extend(images)
            else:
                fake_images.extend(images)

def load_images_from_json_file(json_file):
    if not os.path.isfile(json_file):
        logger.error(f"JSON file does not exist: {json_file}")
        return [], []

    with open(json_file, 'r') as file:
        data = json.load(file)
    real_images = []
    fake_images = []
    load_images_from_json(data, real_images, fake_images)
    return real_images, fake_images

if __name__ == "__main__":
    logger = create_logger('fre.log')
    dataset = ['Celeb-DF-v2']
    for filename in dataset:
        try:
            logger.info(f"Processing dataset: {filename}")
            json_file = f'../preprocessing/dataset_json/{filename}.json'

            if not os.path.isfile(json_file):
                logger.error(f"JSON file does not exist: {json_file}")
                continue

            real_images, fake_images = load_images_from_json_file(json_file)
            logger.info(f"Number of real images: {len(real_images)}")
            logger.info(f"Number of fake images: {len(fake_images)}")

            # Create the output directory if it doesn't exist
            os.makedirs('img', exist_ok=True)

            visualize_average_frequency_spectra(real_images, fake_images, filename)
        except Exception as e:
            logger.error(f"Error processing dataset: {filename}")
            logger.error(f"Error message: {str(e)}")
