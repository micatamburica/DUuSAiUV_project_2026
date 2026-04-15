import os
import math
import keras
import random
import kagglehub
import pandas as pd
from matplotlib import pyplot as plt

from functionalities import detect_lane_change, visualize_frame, display_menu
from data_generation import sequence_data_generator
from data_preparation import create_sequences, fix_image_path

if __name__ == "__main__":

    # Data preparation for testing functionalities
    dataset_path = kagglehub.dataset_download("andy8744/udacity-self-driving-car-behavioural-cloning")
    dataset_name = "self_driving_car_dataset_jungle"

    make_csv = os.path.join(dataset_path, dataset_name, "driving_log.csv")
    df = pd.read_csv(make_csv, header=None, names=['center_path', 'left_path', 'right_path', 'steering_angle', 'throttle', 'reverse', 'speed'])

    while True:

        while True:

            display_menu()
            choice = input("\nEnter your choice (1-4): ").strip()

            if choice == '1':
                FRAME_NUMBER = 9
                V = 0.5
                break
                
            elif choice == '2':
                FRAME_NUMBER = 70
                V = 0.1
                break

            elif choice == '3':
                FRAME_NUMBER = 50
                V = 0.6
                break
            
            elif choice == '4':
                print("Exiting...")
                exit(0)

            else:
                print("Invalid choice. Please enter 1, 2, 3, or 4.")


        # Preparation of the data
        start_index = random.randint(0, len(df) - FRAME_NUMBER)
        random_part = df.iloc[start_index:start_index + FRAME_NUMBER]

        all_sequences = create_sequences(random_part)
        center_sequences = all_sequences[all_sequences['camera'] == 'center'].reset_index(drop=True)

        make_folder = os.path.join(dataset_path, dataset_name)
        center_sequences['image_paths'] = center_sequences['image_paths'].apply(
            lambda paths: [fix_image_path(p, make_folder) for p in paths])
        
        test_data = sequence_data_generator(center_sequences, shuffle=False)
        steps = math.ceil(len(center_sequences) / 16)

        # Model import and prediction
        model = keras.models.load_model("models/steering_model.keras")

        predictions = model.predict(test_data, steps=steps, verbose=0)
        predictions = predictions.flatten()

        plt.ion() 
        fig = plt.figure(figsize=(10, 6)) 

        # Visualization of the frames
        for i in range(len(predictions) - 4):
            
            window = predictions[i:i+5]
            result = detect_lane_change(window)
            visualize_frame(center_sequences, i=i, predicted_angle=predictions[i+4], lane_change=result, speed=V)

        plt.ioff()
        plt.show()