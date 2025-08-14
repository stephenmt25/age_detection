import os
import cv2
import matplotlib.pyplot as plt
import random
import numpy as np

def explore_dataset(dataset_path):
    """
    Explores the UTKFace dataset by displaying a random selection of images
    and their corresponding ages, and shows a histogram of the age distribution.

    Args:
        dataset_path (str): The path to the UTKFace dataset folder.
    """
    if not os.path.isdir(dataset_path):
        print(f"Error: Dataset path not found at '{dataset_path}'")
        return

    image_files = [f for f in os.listdir(dataset_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    total_images = len(image_files)
    print(f"Total number of images found: {total_images}")

    # Extract ages for histogram
    ages = []
    for filename in image_files:
        try:
            age = int(filename.split('_')[0])
            ages.append(age)
        except (ValueError, IndexError):
            print(f"Warning: Could not parse age from filename: {filename}")

    # Create histogram of age distribution
    if ages:
        plt.figure(figsize=(10, 5))
        plt.hist(ages, bins=30, edgecolor='black')
        plt.title('Age Distribution in UTKFace Dataset')
        plt.xlabel('Age')
        plt.ylabel('Number of Images')
        plt.grid(True, alpha=0.3)
        plt.show()

    # Display 3x3 grid of random images
    if total_images < 9:
        print("Error: Not enough images to display a 3x3 grid.")
        return

    random.shuffle(image_files)
    selected_images = image_files[:9]

    plt.figure(figsize=(10, 10))
    for i, filename in enumerate(selected_images):
        try:
            age = int(filename.split('_')[0])
            img_path = os.path.join(dataset_path, filename)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            plt.subplot(3, 3, i + 1)
            plt.imshow(img)
            plt.title(f"Age: {age}")
            plt.axis('off')
        except (ValueError, IndexError):
            print(f"Warning: Could not parse age from filename: {filename}")
            # Create a blank image with a warning message
            blank_img = 255 * np.ones((100, 100, 3), np.uint8)
            cv2.putText(blank_img, "Error", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            plt.subplot(3, 3, i + 1)
            plt.imshow(blank_img)
            plt.title("Error")
            plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # The dataset path is hardcoded to avoid issues with spaces in the path.
    dataset_path = r"C:\Users\stevo\CV PROJECT\UTKFace"
    explore_dataset(dataset_path)