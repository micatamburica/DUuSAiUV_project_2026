import os
import math

from model import create_cnn_lstm_model
from data_generation import sequence_data_generator

def train_model(df_train, df_val, epochs=10, batch_size=16, sequence_length=5):
    """Train the steering angle prediction model."""

    # Get the batches ready for training
    train_gen = sequence_data_generator(df_train, batch_size=batch_size)
    val_gen = sequence_data_generator(df_val, batch_size=batch_size, shuffle=False)

    steps_per_epoch = math.ceil(len(df_train) / batch_size)
    validation_steps = math.ceil(len(df_val) / batch_size)

    # Creating CNN+LSTM model architecture
    model = create_cnn_lstm_model(sequence_length=sequence_length)

    print(f"Epochs chosen to avoid overfitting: {epochs}")
    print(f"Batch size: {batch_size} - Steps per epoch: {steps_per_epoch} - Validation steps: {validation_steps}")
    print("="*120 + "\n")

    model.summary()
    print("="*120 + "\n")
    
    # Model training
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=validation_steps,
        verbose=1
    )
    
    if not os.path.exists('models'):
        os.makedirs('models')

    model.save('models/steering_model.keras')

    print("="*120 + "\n")
    
    return model, history