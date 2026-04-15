import os
import numpy as np
from matplotlib import pyplot as plt

def visualize_frame(center_sequences, sequence_length = 5, i=0, predicted_angle=0.0, lane_change='straight', speed=0.6):

    # Show the last frame in the window
    row = center_sequences.iloc[i + sequence_length-1]

    real_angle = row['steering_angle']

    last_image_path = row['image_paths'][-1]
    img = plt.imread(last_image_path)

    plt.clf()
    
    ax = plt.gca() 
    ax.imshow(img)
 
    color = 'white'
    if (abs(predicted_angle - real_angle)) > 0.2:
        color = 'orange'

    if i is not 0:
        ax.text(img.shape[1]//2, 30, f'{lane_change}', fontsize=60, fontweight='bold', ha='center', color='white')

    info_text = f"{real_angle: .2f} REAL\n{predicted_angle: .2f} PREDICTED"
    ax.text(10, img.shape[0] - 10, info_text, fontsize=18, fontweight='bold',
            va='bottom', bbox=dict(boxstyle='round,pad=0.5', facecolor=color, edgecolor='none', alpha=0.7))
    
    ax.axis('off')
        
    plt.tight_layout()
    plt.draw()
    plt.pause(speed)


def display_menu():
    """Display the interactive menu."""

    os.system('cls' if os.name == 'nt' else 'clear')
    print("\n" + "="*40)
    print("1. Test random image prediction")
    print("2. Test random lane change detection [ fast ⚡ ]")
    print("3. Test random lane change detection [ slow 🐌 ]")
    print("4. Exit")
    print("="*40)


def detect_lane_change(steering_sequence, threshold=0.20):
    """Detect if a lane change occurred in a sequence of steering angles."""
    
    angles = np.array(steering_sequence)
    mean_angle = np.mean(angles)
    
    if mean_angle > threshold:
        return '↗'
    
    elif mean_angle < -threshold:
        return '↖'
    
    else:
        return '↑'
