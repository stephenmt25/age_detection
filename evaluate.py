
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from data_pipeline import create_datasets

# Constants
MODEL_PATH = 'age_estimator_final.h5'
DATASET_PATH = 'C:/Users/stevo/CV PROJECT/UTKFace'
BATCH_SIZE = 32
IMAGE_SIZE = (128, 128)

# 1. Load the trained model
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

# 2. Load the test dataset
print("Loading test dataset...")
_, _, test_ds, _, _, test_size = create_datasets(DATASET_PATH, BATCH_SIZE)

# 3. Evaluate the model
print("Evaluating model...")
evaluation_results = model.evaluate(test_ds, steps=test_size // BATCH_SIZE)
print(f"Final Test Mean Absolute Error: {evaluation_results[1]}")

# 4. Generate predictions
print("Generating predictions...")
predictions = model.predict(test_ds, steps=test_size // BATCH_SIZE)

# Extract true labels from the dataset
true_ages = []
for _, labels in test_ds.take(test_size // BATCH_SIZE):
    true_ages.extend(labels.numpy())
true_ages = np.array(true_ages)

# 5. Create a scatter plot
print("Creating prediction scatter plot...")
plt.figure(figsize=(8, 8))
plt.scatter(true_ages, predictions.flatten())
plt.xlabel("True Age")
plt.ylabel("Predicted Age")
plt.title("True Age vs. Predicted Age")
plt.savefig("prediction_scatter.png")
print("Scatter plot saved as prediction_scatter.png")

# 6. Display random sample images with predictions
print("Displaying sample predictions...")
plt.figure(figsize=(15, 15))
# Get one batch from the test set
for images, labels in test_ds.take(1):
    sample_predictions = model.predict(images)
    for i in range(min(9, len(images))):  # Display up to 9 images
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy())
        plt.title(f"True: {labels[i].numpy()}, Pred: {sample_predictions[i][0]:.2f}")
        plt.axis("off")
plt.savefig("sample_predictions.png")
print("Sample predictions plot saved as sample_predictions.png")
