
import os
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

def parse_function(filename, label):
    image_string = tf.io.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image_resized = tf.image.resize(image_decoded, [128, 128])
    image_normalized = image_resized / 255.0
    return image_normalized, label

def create_datasets(dataset_path, batch_size):
    """
    Builds a complete data processing pipeline for the UTKFace dataset.

    Args:
        dataset_path (str): The path to the UTKFace dataset directory.
        batch_size (int): The batch size for the datasets.

    Returns:
        tuple: A tuple containing the training, validation, and testing datasets,
               and the sizes of each set.
    """
    # 1. Scans the dataset_path and parses all filenames to extract the age.
    image_paths = []
    ages = []
    for filename in os.listdir(dataset_path):
        image_path = os.path.join(dataset_path, filename)
        try:
            age = int(filename.split('_')[0])
            image_paths.append(image_path)
            ages.append(age)
        except (ValueError, IndexError):
            # Ignore files that do not follow the naming convention
            continue

    # 2. Stores the full image file paths and their corresponding age labels in a Pandas DataFrame.
    df = pd.DataFrame({'image_path': image_paths, 'age': ages})

    # 3. Splits the DataFrame into training (70%), validation (15%), and testing (15%) sets.
    train_df, test_val_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(test_val_df, test_size=0.5, random_state=42)

    train_size = len(train_df)
    val_size = len(val_df)
    test_size = len(test_df)

    # 4. Creates three tf.data.Dataset objects from the splits.
    train_ds = tf.data.Dataset.from_tensor_slices((train_df['image_path'].values, train_df['age'].values))
    val_ds = tf.data.Dataset.from_tensor_slices((val_df['image_path'].values, val_df['age'].values))
    test_ds = tf.data.Dataset.from_tensor_slices((test_df['image_path'].values, test_df['age'].values))

    # 5. The training dataset pipeline must include: shuffling, data augmentation, batching, and prefetching.
    train_ds = train_ds.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.shuffle(buffer_size=train_size)
    train_ds = train_ds.map(lambda x, y: (tf.image.random_flip_left_right(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.map(lambda x, y: (tf.image.random_brightness(x, max_delta=0.2), y), num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    # 6. The validation and testing dataset pipelines must only include batching and prefetching.
    val_ds = val_ds.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    test_ds = test_ds.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size)
    test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    # 7. The function should return the three tf.data.Dataset objects and the sizes of each set.
    return train_ds, val_ds, test_ds, train_size, val_size, test_size

if __name__ == '__main__':
    # This is an example of how to use the create_datasets function.
    # Before running this, you need to download the UTKFace dataset and extract it.
    # You can find the dataset here: https://susanqq.github.io/UTKFace/
    # Make sure to place the extracted 'UTKFace' folder in the same directory as this script,
    # or provide the correct path to the dataset.
    
    dataset_path = 'C:/Users/stevo/CV PROJECT/UTKFace'
    batch_size = 32

    if os.path.exists(dataset_path):
        train_dataset, val_dataset, test_dataset, train_count, val_count, test_count = create_datasets(dataset_path, batch_size)
        print(f"Training dataset size: {train_count}")
        print(f"Validation dataset size: {val_count}")
        print(f"Testing dataset size: {test_count}")
        print("Datasets created successfully.")
    else:
        print(f"Dataset not found at path: {dataset_path}")
        print("Please download the UTKFace dataset and place it in the correct directory.")
