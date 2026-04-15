from tensorflow import keras    

def create_cnn_lstm_model(sequence_length=5):
    """CNN+LSTM model architecture inspired by NVIDIA's paper."""

    # Input layer (5 frames of 66x200 RGB images)
    sequence_input = keras.Input(shape=(sequence_length, 66, 200, 3), name='sequence_input')


    # CNN layers (applied to each frame via TimeDistributed)
    conv1 = keras.layers.TimeDistributed(keras.layers.Conv2D(24, (5, 5), strides=(2, 2), activation='relu',
                                padding='valid', name='td_conv1'))(sequence_input)
    
    conv2 = keras.layers.TimeDistributed(keras.layers.Conv2D(36, (5, 5), strides=(2, 2), activation='relu',
                                padding='valid', name='td_conv2'))(conv1)
    
    conv3 = keras.layers.TimeDistributed(keras.layers.Conv2D(48, (5, 5), strides=(2, 2), activation='relu',
                                padding='valid', name='td_conv3'))(conv2)
    
    conv4 = keras.layers.TimeDistributed(keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu',
                                padding='valid', name='td_conv4'))(conv3)
    
    conv5 = keras.layers.TimeDistributed(keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu',
                                padding='valid', name='td_conv5'))(conv4)
    

    # Flatten CNN features from each frame
    flatten = keras.layers.TimeDistributed(keras.layers.Flatten(), name='td_flatten')(conv5)


    # LSTM layers for temporal processing
    lstm1 = keras.layers.LSTM(64, return_sequences=True, name='lstm1')(flatten)
    lstm_dropout1 = keras.layers.Dropout(0.3, name='lstm_dropout1')(lstm1)

    lstm2 = keras.layers.LSTM(32, return_sequences=False, name='lstm2')(lstm_dropout1)
    lstm_dropout2 = keras.layers.Dropout(0.3, name='lstm_dropout2')(lstm2)


    # Fully connected layers
    fc1 = keras.layers.Dense(50, activation='relu', name='fc1')(lstm_dropout2)
    fc_dropout1 = keras.layers.Dropout(0.5, name='fc_dropout1')(fc1)

    fc2 = keras.layers.Dense(10, activation='relu', name='fc2')(fc_dropout1)


    # Output layer (single steering angle prediction)
    output_steering = keras.layers.Dense(1, name='output_steering')(fc2)

    # Create the model
    model = keras.Model(inputs=[sequence_input], outputs=[output_steering], name='CNN_LSTM_Steering_Prediction')

    # Compile the model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    return model