import cv2
import numpy as np

def single_image_preprocessing(image_path):
    """"Load and preprocess a single image, by resizing and normalizing it."""

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"WARNING: Could not load image at path {image_path}")
        return None
    
    # Convert the image from BGR to RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Crop the image, removing unnecessary information (sky and car hood)
    image = image[56:-16, :]

    # Resize to target size (NVIDIA architecture input size)
    image = cv2.resize(image, (200, 66))

    # Normalize to [-1, 1] range
    image = image.astype(np.float32)
    image = (image / 127.5) - 1.0
    
    return image


def sequence_data_generator(df_sequences, batch_size=16, shuffle=True):
    """Generate batches of sequences for training."""
    
    num_samples = len(df_sequences)
    
    while True:
        
        # Shuffle at the beginning of each epoch
        if shuffle:
            df_sequences = df_sequences.sample(frac=1).reset_index(drop=True)
        
        # Generate batches
        for offset in range(0, num_samples, batch_size):
            
            batch_df = df_sequences.iloc[offset:offset + batch_size]
            
            batch_images = []
            batch_angles = []
            
            # Process each sequence in the batch
            for _, row in batch_df.iterrows():
                
                sequence_images = []
                
                # Load each image in the sequence
                for img_path in row['image_paths']:
                    img = single_image_preprocessing(img_path)
                    
                    if img is not None:
                        sequence_images.append(img)
                    else:
                        break
                
                # Only add if all images loaded successfully
                if len(sequence_images) == len(row['image_paths']):
                    batch_images.append(sequence_images)
                    batch_angles.append(row['steering_angle'])
            
            # Yield batch
            if len(batch_images) > 0:
                yield np.array(batch_images), np.array(batch_angles)