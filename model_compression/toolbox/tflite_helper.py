import tensorflow as tf
import numpy as np
from typing import Optional
from sklearn.metrics import confusion_matrix, classification_report
from .sequence_gen import Sequence

class TFLiteQuantModel:
    def __init__(self, model : tf.keras.Sequential, paramaters : dict):
        self.model = None
        self.converter = tf.lite.TFLiteConverter.from_keras_model(model)
        self.converter.optimizations = [paramaters["optimizations"]]
        self.converter.target_spec.supported_ops = [paramaters["opset"]]
        if paramaters["representative_dataset_gen"] is not None:
            self.converter.representative_dataset = self.__create_rep_datagen(paramaters["representative_dataset_gen"])
        self.interpreter = None
        self.quant_type = "float32"
        
    def __create_rep_datagen(self, rep_seq : Sequence):
        def rep_datagen():
            for batch in rep_seq:
                for item in batch[0]:
                    shape = (1,) + item.shape
                    new_item = np.reshape(item.astype(np.float32), shape)

                    if new_item.shape != (1, 16, 10, 64):
                        print("Something is wrong!")
                        print(new_item.shape)

                    yield [new_item]

        return rep_datagen
            
    def convert(self):
        try:
            self.model = self.converter.convert()
        except:
            raise Exception("Conversion unsuccessful.")
            
    def save(self, filename : str, directory : str):
        if self.model is not None:
            with open(directory + filename, 'wb') as f:
                f.write(self.model)
        print("Saved to " + directory + filename)
                
    def evaluate(self, val_data : Sequence) -> dict: 
        if self.model is not None:
            if self.interpreter is None:
                self.interpreter = tf.lite.Interpreter(model_content=self.model)
                self.interpreter.allocate_tensors()

            input_tensor_index = self.interpreter.get_input_details()[0]["index"]
            output_tensor_index = self.interpreter.get_output_details()[0]["index"]

            preds = []
            trues = []
            pred_prob = []

            print("Evaluating .tflite model. Please wait...")

            for batch in val_data: #batch[0]: samples, batch[1]: labels
                data = batch[0]
                labels = batch[1]
                b_size = len(labels)

                for i in range(b_size):
                    inp = data[i]
                    shape = (1,) + inp.shape
                    inp = np.reshape(inp, shape)

                    y_true = labels[1]

                    if self.quant_type == "float32":
                        inp = inp.astype(np.float32)
                    if self.quant_type == "int8":
                        inp = inp.astype(np.int8)
                    if self.quant_type == "int16":
                        inp = inp.astype(np.int16)

                    self.interpreter.set_tensor(input_tensor_index, inp)
                    self.interpreter.invoke()

                    y_pred = self.interpreter.get_tensor(output_tensor_index)
                    y_pred_class = np.argmax(y_pred, axis=-1)
                    pred_prob.append(np.amax(y_pred, axis=-1))
                    trues.append(y_true)
                    preds.append(y_pred_class)

            y_true = np.array(trues)
            y_pred_prob = np.array(pred_prob)
            y_pred = np.array(preds)
            report = classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)

            return report
        else:
            raise Exception("Model not defined!")
            
