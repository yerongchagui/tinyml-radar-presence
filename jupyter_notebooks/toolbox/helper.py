import tempfile
import zipfile
import os
import numpy as np
import tensorflow as tf

def get_gzipped_model_size(filepath : str):
    # It returns the size of the gzipped model in kilobytes.
    _, zipped_file = tempfile.mkstemp('.zip')
    with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
        f.write(filepath)

    return os.path.getsize(zipped_file)/1000

def save(filepath : str, model : tf.keras.Model):
    with open(filepath, 'wb') as f:
        f.write(model)
    print("Saved to " + filepath)

def evaluate_tflite(path : str, X_test : np.ndarray, y_test : np.ndarray,  verbose : bool = True):
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()

    input_tensor_index = interpreter.get_input_details()[0]["index"]
    output_tensor_index = interpreter.get_output_details()[0]["index"]

    y_pred = []
    y_pred_prob = []

    if verbose:
        print("Evaluating .tflite model. Please wait...")

    for i in range(len(X_test)):
        interpreter.set_tensor(input_tensor_index, np.array([X_test[i],]))
        interpreter.invoke()
        y_pred.append(np.argmax(interpreter.get_tensor(output_tensor_index), axis=-1))
        y_pred_prob.append(np.amax(interpreter.get_tensor(output_tensor_index), axis=-1))

    y_pred = np.asarray(y_pred)
    acc = 0
    for i in range(y_test.shape[0]):
        t = (y_test[i] == y_pred[i])[0]
        if t == True:
            acc += 1
    acc /= y_test.shape[0]

    return acc
