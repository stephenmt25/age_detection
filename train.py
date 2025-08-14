
import tensorflow as tf
import matplotlib.pyplot as plt
from data_pipeline import create_datasets
from model import build_model

# 1. Define constants
IMAGE_SIZE = (128, 128)  # Corrected to match data_pipeline
INPUT_SHAPE = (*IMAGE_SIZE, 3)
BATCH_SIZE = 32
EPOCHS_PHASE_1 = 10
EPOCHS_PHASE_2 = 15
DATASET_PATH = 'C:/Users/stevo/CV PROJECT/UTKFace'

# 2. Call create_datasets
(train_ds, val_ds, test_ds, 
 train_size, val_size, test_size) = create_datasets(DATASET_PATH, BATCH_SIZE)

# 3. Call build_model
model = build_model(INPUT_SHAPE)

# 4. Phase 1 Training
print("--- Starting Phase 1 Training ---")
history_phase1 = model.fit(
    train_ds,
    epochs=EPOCHS_PHASE_1,
    validation_data=val_ds,
    steps_per_epoch=train_size // BATCH_SIZE,
    validation_steps=val_size // BATCH_SIZE
)

# 5. Phase 2 Fine-Tuning
print("--- Starting Phase 2 Fine-Tuning ---")
model.trainable = True

# 6. Re-compile the model with a lower learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='mean_absolute_error',
              metrics=['mean_absolute_error'])

# 7. Continue training
history_phase2 = model.fit(
    train_ds,
    epochs=EPOCHS_PHASE_1 + EPOCHS_PHASE_2, # Total epochs
    initial_epoch=EPOCHS_PHASE_1, # Start from the end of phase 1
    validation_data=val_ds,
    steps_per_epoch=train_size // BATCH_SIZE,
    validation_steps=val_size // BATCH_SIZE
)

# 8. Save the final model
model.save('age_estimator_final.h5')

# 9. Generate and save a plot
mae = history_phase1.history['mean_absolute_error'] + history_phase2.history['mean_absolute_error']
val_mae = history_phase1.history['val_mean_absolute_error'] + history_phase2.history['val_mean_absolute_error']

plt.figure(figsize=(8, 8))
plt.plot(range(EPOCHS_PHASE_1 + EPOCHS_PHASE_2), mae, label='Training MAE')
plt.plot(range(EPOCHS_PHASE_1 + EPOCHS_PHASE_2), val_mae, label='Validation MAE')
plt.legend(loc='upper right')
plt.title('Training and Validation MAE')
plt.savefig('training_history.png')
