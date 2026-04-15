import math
import numpy as np
from data_generation import sequence_data_generator

def evaluate_model(model, df_test, batch_size=16):
    """Evaluate the trained model on the test set."""

    STEERING_MIN = -1.0
    STEERING_MAX = 1.0
    STEERING_RANGE = STEERING_MAX - STEERING_MIN
    
    # Get the batches ready for evaluation
    test_gen = sequence_data_generator(df_test, batch_size=batch_size, shuffle=False)
    test_steps = math.ceil(len(df_test) / batch_size)
    
    # Evaluate
    test_loss, test_mae = model.evaluate(test_gen, steps=test_steps, verbose=1)
    
    # Print results
    print(f"Test MSE:       {test_loss:.6f}")
    print(f"Test MAE:       {test_mae:.6f}      ({(test_mae/STEERING_RANGE)*100:.2f}% of range)")
    print(f"Test RMSE:      {np.sqrt(test_loss):.6f}     ({(np.sqrt(test_loss)/STEERING_RANGE)*100:.2f}% of range)\n")

    # Predictions and error analysis
    test_gen_pred = sequence_data_generator(df_test, batch_size=batch_size, shuffle=False)
    predictions = model.predict(test_gen_pred, steps=test_steps, verbose=1)
    predictions = predictions.flatten()
    
    real_angles = df_test['steering_angle'].values[:len(predictions)]
    
    # Calculate errors
    errors = predictions - real_angles
    abs_errors = np.abs(errors)
    
    # Print results
    print(f"\nTotal predictions:      {len(predictions)}")
    print(f"Mean Error (bias):        {np.mean(errors):+.6f}        ({(np.mean(errors)/STEERING_RANGE)*100:+.2f}% of range)")
    print(f"Std of Errors:            {np.std(errors):.6f}          ({(np.std(errors)/STEERING_RANGE)*100:.2f}% of range)")
    print(f"Min Error:                {np.min(errors):+.6f}         ({(np.min(errors)/STEERING_RANGE)*100:+.2f}% of range)")
    print(f"Max Error:                {np.max(errors):+.6f}         ({(np.max(errors)/STEERING_RANGE)*100:+.2f}% of range)")
    print(f"Median Absolute Error:    {np.median(abs_errors):.6f}   ({(np.median(abs_errors)/STEERING_RANGE)*100:.2f}% of range)")
 