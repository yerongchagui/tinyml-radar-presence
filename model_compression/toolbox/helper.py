import yaml
import tempfile
import numpy as np
import zipfile
from sklearn.metrics import classification_report
import os
import tensorflow as tf

from IPython.display import clear_output
from .sequence_gen import Sequence
from typing import Any

def load_yaml(file_path: str) -> Any:
    """
    Open .yaml-file and return values as dictionary.

    :param file_path: Path to file
    :return: Data as dictionary or list
    """
    with open(file_path, mode="r") as yaml_file:
        yaml_data: Dict = yaml.safe_load(yaml_file)
    return yaml_data

def get_gzipped_model_size(filepath):
    # It returns the size of the gzipped model in kilobytes.
    _, zipped_file = tempfile.mkstemp('.zip')
    with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
        f.write(filepath)

    return os.path.getsize(zipped_file)/1000

def tf_model_evaluate(model : tf.keras.Sequential, val_data : Sequence) -> dict:
    preds = []
    trues = []
    pred_prob = []
    count = 0
    
    # Perform inference over validation dataset and collect results
    for data in val_data:   
        y_pred = model.predict(data[0], verbose=0)
        clear_output(wait=True)
        count += 1
        print("Progress: " + str(count) + "/" + str(len(val_data)))
        
        y_pred_class = np.argmax(y_pred, axis=-1)
        y_true = data[1]
        pred_prob.append(np.amax(y_pred, axis=-1))
        trues.append(y_true)
        preds.append(y_pred_class)
        
    y_true = np.concatenate(trues, axis=0)
    y_pred_prob = np.concatenate(pred_prob, axis=0)
    y_pred = np.concatenate(preds, axis=0)
    
    report = classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)
    return report
