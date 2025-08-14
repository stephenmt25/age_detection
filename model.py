
import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model

def build_model(input_shape):
    """
    Builds and compiles a regression model using ResNet50V2 as a base.

    Args:
        input_shape (tuple): The shape of the input images (height, width, channels).

    Returns:
        tf.keras.Model: The compiled Keras model.
    """
    # 1. Use ResNet50V2 as the base model
    base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=input_shape)

    # 2. Set the base model's layers to be non-trainable
    base_model.trainable = False

    # 3. Add a custom regression head
    head = base_model.output
    head = GlobalAveragePooling2D()(head)
    head = Dropout(0.5)(head)
    head = Dense(1, activation='linear')(head)

    # 4. Create the final Model
    model = Model(inputs=base_model.input, outputs=head)

    # 5. Compile the model
    model.compile(optimizer='adam',
                  loss='mean_absolute_error',
                  metrics=['mean_absolute_error'])

    # 6. Return the compiled model
    return model

if __name__ == '__main__':
    # This is an example of how to use the build_model function.
    input_shape = (128, 128, 3)
    model = build_model(input_shape)
    model.summary()
