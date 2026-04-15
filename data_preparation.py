import os
import numpy as np
import pandas as pd


def fix_image_path(old_path, dataset_folder):
    """Fix the image path by joining the dataset folder with the image filename."""
    
    filename = os.path.basename(old_path)
    return os.path.join(dataset_folder, 'IMG', filename)


def create_sequences(df, sequence_length=5, dataset_name='unknown'):
    """Create sequences from CSV data for all three cameras separately."""
    
    sequences = []
    
    cameras = ['center', 'left', 'right']
    camera_columns = {'center': 'center_path', 'left': 'left_path', 'right': 'right_path'}
    
    for camera_name in cameras:
        for i in range(len(df) - sequence_length + 1):
            
            image_paths = []
            
            # Collect image paths for this camera across the sequence
            for j in range(sequence_length):
                image_paths.append(df.iloc[i + j][camera_columns[camera_name]])
            
            # Target steering angle from the last frame
            base_angle = df.iloc[i + sequence_length - 1]['steering_angle']
            
            sequences.append({
                'image_paths': image_paths,
                'steering_angle': base_angle,
                'dataset': dataset_name,
                'camera': camera_name
            })
    
    df_sequences = pd.DataFrame(sequences)

    print(f"Created {len(df_sequences)} sequences from {len(df)} frames ({dataset_name}).")
    
    return df_sequences


def camera_steering_correction(df_sequences, steering_correction=0.2):
    """Apply steering angle corrections for left and right cameras and clip to valid range."""
    
    df_corrected = df_sequences.copy()
    
    # Apply corrections based on camera type
    for idx, row in df_corrected.iterrows():
        
        original_angle = row['steering_angle']
        camera = row['camera']
        
        if camera == 'center':
            corrected_angle = original_angle
        elif camera == 'left':
            corrected_angle = original_angle - steering_correction
        elif camera == 'right':
            corrected_angle = original_angle + steering_correction
        else:
            corrected_angle = original_angle
        
        # Clip to valid range [-1.0, 1.0]
        clipped_angle = np.clip(corrected_angle, -1.0, 1.0)
        
        df_corrected.at[idx, 'steering_angle'] = clipped_angle
    
    return df_corrected


def data_combination(df_make_seq, df_jungle_seq, dataset_path):
    """Combine sequence data into one dataset and fix image paths."""
    
    # Fix paths for Make dataset
    make_folder = os.path.join(dataset_path, "self_driving_car_dataset_make")
    df_make_seq['image_paths'] = df_make_seq['image_paths'].apply(
        lambda paths: [fix_image_path(p, make_folder) for p in paths])

    # Fix paths for Jungle dataset
    jungle_folder = os.path.join(dataset_path, "self_driving_car_dataset_jungle")
    df_jungle_seq['image_paths'] = df_jungle_seq['image_paths'].apply(
        lambda paths: [fix_image_path(p, jungle_folder) for p in paths])
    
    # Combine the two dataframes
    df_combined = pd.concat([df_make_seq, df_jungle_seq], ignore_index=True)
    
    # Shuffle
    df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Created combined dataset of {len(df_combined)} sequences.")
    
    return df_combined


def balance_dataset(df, steering_correction=0.2, fraction=0.1):
    """Balance the dataset by reducing over-represented zero steering angle."""

    # Separate the bias zero values steering
    zero_steering = df[df['steering_angle'] == 0]
    left_steering = df[df['steering_angle'] == steering_correction]
    right_steering = df[df['steering_angle'] == -steering_correction]
    non_zero_steering = df[(df['steering_angle'] != 0) & (df['steering_angle'] != steering_correction) & (df['steering_angle'] != -steering_correction)]

    # Sample fraction of zero-steering data (frac can be adjusted)
    zero_steering_sampled = zero_steering.sample(frac=fraction, random_state=42)
    left_steering_sampled = left_steering.sample(frac=fraction, random_state=42)
    right_steering_sampled = right_steering.sample(frac=fraction, random_state=42)

    # Combine the non-zero-steering data with sampled zero-steering data and shuffle
    df_balanced = pd.concat([non_zero_steering, zero_steering_sampled, left_steering_sampled, right_steering_sampled], ignore_index=True)
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"New dataset of {len(df_balanced)} sequences. Removed {len(df) - len(df_balanced)} ({(len(df) - len(df_balanced)) / len(df) * 100:.2f}%) sequences for balancing.")

    return df_balanced


def split_dataset(df, train_size=0.7, val_size=0.15, test_size=0.15):
    """Split the dataset into training, validation and test sets."""

    # Shuffle the dataset before splitting
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    test_split = int(len(df) * test_size)
    val_split = int(len(df) * (test_size + val_size))

    # Split into tree sets
    df_test = df[:test_split]
    df_val = df[test_split:val_split]
    df_train = df[val_split:]

    # Reset indices
    df_test = df_test.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_train = df_train.reset_index(drop=True)

    print(f"Dataset split into {len(df_train)} ({train_size*100:.1f}%) training, {len(df_val)} ({val_size*100:.1f}%) validation and {len(df_test)} ({test_size*100:.1f}%) test sequences.")

    return df_train, df_val, df_test
