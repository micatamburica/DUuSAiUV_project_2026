## **SELF-DRIVING CAR STEERING ANGLE PREDICTION**

The goals of the project is to predict the steering angle of the vehicle based on series of images and detect lane change based on the sequence of steering angles.

* Recordings of the front cameras are available - left, central, right (in the form of series of images).
* When determining the steering angle, use the combined CNN+LSTM architecture (CNN for extracting features from each image, and LSTM for processing a series of these features).
* The predicted steering angle must be shown on the image (in the format of your choice).
* Detect lane change based on predicted steering angle. In case of lane change detection, signal a warning (signaling is done as per your choice: visual, sound or some third way).
* Implement the solution in Python, using TensorFlow/Keras and other Python modules necessary (for drawing, image processing, etc.).

---
### **The model training can be done with the given command:   <ins>python main.py</ins>**
1. Download/load the dataset
2. Prepare the dataset for training
3. Train the CNN-LSTM model
5. Evaluate the model
_More on main.py can be found down at the main.py section_

### **The model visual testing can be done with the given command:   <ins>python test.py</ins>**
1. Test random image prediction
2. Test random lane change detection

---
**EXPECTED OUTPUT FROM MAIN.PY:**

Created 11778 sequences from 3930 frames (make).<br />
Created 10200 sequences from 3404 frames (jungle).<br />
Created combined dataset of 21978 sequences.<br />
New dataset of 9228 sequences. Removed 12750 (58.01%) sequences for balancing.<br />
Dataset split into 6460 (70.0%) training, 1384 (15.0%) validation and 1384 (15.0%) test sequences.<br />

Epochs chosen to avoid overfitting: 10<br />
Batch size: 16 - Steps per epoch: 404 - Validation steps 87<br />

Model: "CNN_LSTM_Steering_Prediction"
| Layer (type) | Output Shape | Param # |
|-------------|--------------|---------|
| sequence_input (InputLayer) | (None, 5, 66, 200, 3) | 0 |
| time_distributed (TimeDistributed) | (None, 5, 31, 98, 24) | 1,824 |
| time_distributed_1 (TimeDistributed) | (None, 5, 14, 47, 36) | 21,636 |
| time_distributed_2 (TimeDistributed) | (None, 5, 5, 22, 48) | 43,248 |
| time_distributed_3 (TimeDistributed) | (None, 5, 3, 20, 64) | 27,712 |
| time_distributed_4 (TimeDistributed) | (None, 5, 1, 18, 64) | 36,928 |
| td_flatten (TimeDistributed) | (None, 5, 1152) | 0 |
| lstm1 (LSTM) | (None, 5, 64) | 311,552 |
| lstm_dropout1 (Dropout) | (None, 5, 64) | 0 |
| lstm2 (LSTM) | (None, 32) | 12,416 |
| lstm_dropout2 (Dropout) | (None, 32) | 0 |
| fc1 (Dense) | (None, 50) | 1,650 |
| fc_dropout1 (Dropout) | (None, 50) | 0 |
| fc2 (Dense) | (None, 10) | 510 |

Total params: 457,487 (1.75 MB)<br />
Trainable params: 457,487 (1.75 MB)<br />
Non-trainable params: 0 (0.00 B)<br />

Training History (10 Epochs)
| Epoch | Train Loss | Train MAE | Val Loss | Val MAE | Time |
|-------|------------|-----------|----------|---------|------|
| 1 | 0.1692 | 0.3220 | 0.1070 | 0.2606 | 164s |
| 2 | 0.1155 | 0.2652 | 0.1053 | 0.2536 | 134s |
| 3 | 0.1000 | 0.2459 | 0.0997 | 0.2475 | 132s |
| 4 | 0.0889 | 0.2317 | 0.0919 | 0.2401 | 142s |
| 5 | 0.0803 | 0.2199 | 0.0795 | 0.2224 | 140s |
| 6 | 0.0741 | 0.2123 | 0.0811 | 0.2251 | 132s |
| 7 | 0.0699 | 0.2062 | 0.0685 | 0.2071 | 112s |
| 8 | 0.0643 | 0.1989 | 0.0754 | 0.2176 | 97s |
| 9 | 0.0644 | 0.1965 | 0.0788 | 0.2202 | 102s |
| 10 | 0.0583 | 0.1878 | 0.0757 | 0.2160 | 96s |

| Metric | Value | % of Range |
|--------|-------|------------|
| **MSE** | 0.0703 | - |
| **MAE** | 0.2044 | 10.22% |
| **RMSE** | 0.2651 | 13.26% |

| Metric | Value | % of Range |
|--------|-------|------------|
| Mean Error (bias) | +0.0732 | +3.66% |
| Std of Errors | 0.2548 | 12.74% |
| Median Absolute Error | 0.1611 | 8.06% |
| Min Error | -0.6968 | -34.84% |
| Max Error | +1.0160 | +50.80% |


**EXPECTED OUTPUT FROM TEST.PY:**

1. Test random image prediction
2. Test random lane change detection [ fast ⚡ ]
3. Test random lane change detection [ slow 🐌 ]
4. Exit

Enter your choice (1-4): 

---

### Project Structure
```
project/<br />
│<br />
├── **main.py**                 MAIN TRAINING PIPELINE<br />
├──── data_preparation.py       Data loading and preprocessing<br />
├──── data_generation.py        Data generator for training<br />
├──── model.py                  CNN-LSTM model architecture<br />
├──── train.py                  Training logic<br />
├──── evaluate.py               Model evaluation<br />
├──── plot_handling.py          Visualization utilities<br />
├── **test.py**                 TESTING AND VISUALIZATION<br />
├── functionalities.py          Lane change detection algorithm<br />
│<br />
├── graphs/<br />
│    └─ training_history.png    Training curves (generated)<br />
├── histograms/<br />
│    └─ steering_angle.png      Data distribution (generated)<br />
├── graphs/<br />
     └─ steering_model.keras    Trained model (generated)<br />
```

### Dependencies
- Python 3.8+
- tensorflow >= 2.10.0
- keras >= 2.10.0
- numpy >= 1.23.0
- pandas >= 1.5.0
- opencv-python >= 4.7.0
- matplotlib >= 3.6.0
- kagglehub >= 0.1.0

---

### main.py

  #### 1. DOWNLOAD THE DATASET FROM KAGGLE AND LOAD INTO DATAFRAMES
  
  dataset_path = kagglehub.dataset_download("andy8744/udacity-self-driving-car-behavioural-cloning")

  There can be problems when downloading the dataset, in which case go to the site:<br />
  https://www.kaggle.com/datasets/andy8744/udacity-self-driving-car-behavioural-cloning and download the dataset manually.<br />
  Setup the dataset_path to where the dataset is stored on your computer, example:<br />
  dataset_path = r"C:\Users\.cache\kagglehub\datasets\andy8744\udacity-self-driving-car-behavioural-cloning\versions\1".<br />

  #### 2. DATA PREPARATION

  ##### 2.1. create sequences (recommended SEQUENCE_LENGTH = 5 [5 - 10], not too short, not too long)

  before:<br />
  ```
  center_path             left_path               right_path              steering_angle  throttle    reverse     speed<br />
  ..\IMG\center_1.jpg     ..\IMG\left_1.jpg       ..\IMG\right_1.jpg      0               0           0           0.000013<br />
  ```

  after:<br />
  ```
  image_paths                                                                                 steering_angle  dataset     camera<br />
  [..\center_1.jpg, ..\center_2.jpg, ..\center_3.jpg, ..\center_4.jpg, ..\center_5.jpg]       0.00            make        center<br />
  [..\left_1.jpg, ..\left_2.jpg, ..\left_3.jpg, ..\left_4.jpg, ..\lef_t5.jpg]                 0.00            make        left<br />
  [..\right_1.jpg, ..\right_2.jpg, ..\right_3.jpg, ..\right_4.jpg, ..\right_5.jpg]            0.00            make        right<br />
  ```

  ##### 2.2. camera steering correction (recommended STEERING_CORRECTION = 0.2 [0.2 - 0.4], showed best results)

  center camera:      steering_angle<br />
  left camera:        steering_angle -= STEERING_CORRECTION<br />
  right camera:       steering_angle += STEERING_CORRECTION<br />

  ##### 2.3. data combination (and fixing the image path)

  combined dataset = make dataset + jungle dataset [shuffled]

  ##### 2.4. balance dataset

  because of dominant 0 values (around 65%), fraction (90%) of it is removed to balance the dataset

  ##### 2.5. split dataset

  training:   70%<br />
  validation: 15%<br />
  evaluation: 15%<br />


  #### 3. MODEL TRAINING

  ##### 3.1. sequence data generator

  before: Data Frame (? rows x 5 columns)<br />
  ```
  image_paths                                                                                 steering_angle  dataset     camera<br />
  [..\center_1.jpg, ..\center_2.jpg, ..\center_3.jpg, ..\center_4.jpg, ..\center_5.jpg]       0.00            make        center<br />
  ```

  after: Numpy Arrays (tuple Images, Angles)<br />
  Images.shape = (batch size, sequence length, height, width, RGB)    - 15 batches of 5 images (preprocessed)<br />
  Angles.shape = (batch size, )                                       - 16 numbers<br />

  Every single image goes thru preprocessing as such:<br />
      BGR -> RGB color format             (blue, green, red to reverse)<br />
      Rows 1 thru 160 -> Rows 56 thru 144 (cutting out the sky and car hub)<br />
      Resize [320, 160] -> [200, 66]      (image size)<br />
      Normalize [0, 255] -> [-1, 1]       (range of values)<br />

  ##### 3.2. CNN + LSTM model

  Input: (batch, 5, 66, 200, 3)  # 5 frames of 66x200 RGB images

  TimeDistributed CNN Layers:<br />
  ├── Conv2D(24, 5x5, stride=2) → ReLU<br />
  ├── Conv2D(36, 5x5, stride=2) → ReLU<br />
  ├── Conv2D(48, 5x5, stride=2) → ReLU<br />
  ├── Conv2D(64, 3x3, stride=1) → ReLU<br />
  ├── Conv2D(64, 3x3, stride=1) → ReLU<br />
  └── Flatten<br />

  LSTM Layers:<br />
  ├── LSTM(64, return_sequences=True) → Dropout(0.3)<br />
  └── LSTM(32, return_sequences=False) → Dropout(0.3)<br />

  Fully Connected Layers:<br />
  ├── Dense(50) → ReLU → Dropout(0.5)<br />
  ├── Dense(10) → ReLU<br />
        └── Dense(1)  # Output: steering angle<br />

  Output: Single steering angle prediction (-1.0 to 1.0)
