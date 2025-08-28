# Age Estimation from Facial Images

This project implements a deep learning model to estimate the age of a person from a facial image. It uses the UTKFace dataset and a pre-trained ResNet50V2 model for transfer learning.

## Features

*   **Model**: Utilizes ResNet50V2, pre-trained on ImageNet, as the base for feature extraction.
*   **Data Pipeline**: Efficient data loading and preprocessing using `tf.data`.
*   **Data Augmentation**: Includes random flipping and brightness adjustments to improve model generalization.
*   **Two-Phase Training**:
    1.  Trains a custom regression head with the base model's layers frozen.
    2.  Fine-tunes the entire model with a lower learning rate.
*   **Evaluation**: Scripts to evaluate the model's Mean Absolute Error (MAE) on a test set and visualize predictions.

## Dataset

This project uses the [UTKFace dataset](https://susanqq.github.io/UTKFace/), which contains over 20,000 face images with annotations for age, gender, and ethnicity. For this project, only the age label is used.

## File Structure

*   [`data_pipeline.py`](data_pipeline.py): Contains functions to load, preprocess, and augment the image data from the UTKFace dataset.
*   [`model.py`](model.py): Defines the ResNet50V2-based regression model architecture.
*   [`train.py`](train.py): The main script to execute the two-phase model training process. It saves the final model and a plot of the training history.
*   [`evaluate.py`](evaluate.py): Loads the trained model to evaluate its performance on the test set and generates prediction visualizations.
*   [`explore_data.py`](explore_data.py): A utility script to visualize random images from the dataset and plot the age distribution.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-directory>
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install tensorflow pandas scikit-learn matplotlib opencv-python
    ```

3.  **Download the Dataset:**
    *   Download the UTKFace dataset from [here](https://susanqq.github.io/UTKFace/).
    *   Extract the `UTKFace` folder.
    *   Update the `DATASET_PATH` variable in [`train.py`](train.py), [`evaluate.py`](evaluate.py), and [`explore_data.py`](explore_data.py) to point to the location of the `UTKFace` folder on your local machine.

## Usage

1.  **Explore the Data (Optional):**
    Run the `explore_data.py` script to get a feel for the dataset.
    ```bash
    python explore_data.py
    ```

2.  **Train the Model:**
    Execute the training script. This will start the two-phase training process and save the final model as `age_estimator_final.h5` and the training history plot as `training_history.png`.
    ```bash
    python train.py
    ```

3.  **Evaluate the Model:**
    After training is complete, run the evaluation script. This will print the final Mean Absolute Error on the test set and save two plots: `prediction_scatter.png` and `sample_predictions.png`.
    ```bash
    python evaluate.py
    ```

## Results

The training and evaluation scripts will produce the following output files:

*   `age_estimator_final.h5`: The trained and saved Keras model.
*   `training_history.png`: A plot showing the Training and Validation Mean Absolute Error over epochs.
*   `prediction_scatter.png`: A scatter plot comparing the model's predicted ages to the true ages.
*   `sample_predictions.png`: A grid of sample images from the test set with