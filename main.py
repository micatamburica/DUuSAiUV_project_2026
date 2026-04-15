import os
import kagglehub
import pandas as pd

from train import train_model
from evaluate import evaluate_model
from plot_handling import graph_training_history
from data_preparation import create_sequences, camera_steering_correction, data_combination, balance_dataset, split_dataset


if __name__ == "__main__":

    # DOWNLOAD THE DATASET FROM KAGGLE AND LOAD INTO DATAFRAMES
    dataset_path = kagglehub.dataset_download("andy8744/udacity-self-driving-car-behavioural-cloning")

    make_path = os.path.join(dataset_path, "self_driving_car_dataset_make", "driving_log.csv")
    jungle_path = os.path.join(dataset_path, "self_driving_car_dataset_jungle", "driving_log.csv")

    df_make = pd.read_csv(make_path, header=None, names=['center_path', 'left_path', 'right_path', 'steering_angle', 'throttle', 'reverse', 'speed'])
    df_jungle = pd.read_csv(jungle_path, header=None, names=['center_path', 'left_path', 'right_path', 'steering_angle', 'throttle', 'reverse', 'speed'])


    # DATA PREPARATION (create sequences -> steering correction -> combine datasets -> balance dataset -> split dataset)
    SEQUENCE_LENGTH = 5      
    STEERING_CORRECTION = 0.2

    df_make_sequences = create_sequences(df_make, sequence_length=SEQUENCE_LENGTH, dataset_name='make')
    df_jungle_sequences = create_sequences(df_jungle, sequence_length=SEQUENCE_LENGTH, dataset_name='jungle')

    df_make_sequences = camera_steering_correction(df_make_sequences, steering_correction=STEERING_CORRECTION)
    df_jungle_sequences = camera_steering_correction(df_jungle_sequences, steering_correction=STEERING_CORRECTION)

    df_combined = data_combination(df_make_sequences, df_jungle_sequences, dataset_path)

    df_balanced = balance_dataset(df_combined, steering_correction=STEERING_CORRECTION)

    df_train, df_val, df_test = split_dataset(df_balanced)
    

    # MODEL TRAINING
    EPOCHS = 10
    BATCH_SIZE = 16
        
    model, history = train_model(df_train, df_val, epochs=EPOCHS, batch_size=BATCH_SIZE, sequence_length=SEQUENCE_LENGTH)

    graph_training_history(history)

    evaluate_model(model, df_test)