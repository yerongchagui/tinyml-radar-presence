from typing import Dict, Any
import argparse
import logging
import sklearn
import matplotlib.pyplot as plt
import sklearn.metrics
import json
from sklearn.metrics import confusion_matrix, classification_report
# from callbacks import CustomCallback, MLH_Callback, ConfusedCallback
import seaborn as sns
import pandas as pd
import datetime
from IPython.display import clear_output
import tensorflow as tf
import yaml 
import os
import numpy as np
import pathlib
import tempfile
import zipfile

class Sequence(tf.keras.utils.Sequence):
    def __init__(self, parent_path, preprocess_func, batch_size=64, classification=False, l_count=4, seq_len=8, seq_step=1, test=False, reshape=False):
        self.preprocess_func = preprocess_func
        self.seq_len = seq_len
        self.seq_step = seq_step
        self.batch_size = batch_size
        self.reshape = reshape
        self.data, self.labels = self.load_raw_data(parent_path, classification=classification, test=test, l_count=l_count, sequence=seq_len)
        self.seq_lens = [s.shape[0] - int(seq_len * seq_step) for s in self.data]
        self.shuffler = np.random.permutation(np.array(self.seq_lens).sum())
    
    def __len__(self):
        return np.array(self.seq_lens).sum() // self.batch_size

    @classmethod
    def __generate_identifier(cls, parent_path, batch_size, classification, l_count, seq_len, seq_step, test, reshape):
        string = f"{parent_path}-{batch_size}-{classification}-{l_count}-{seq_len}-{seq_step}-{reshape}"
        identifier = hashlib.md5(string.encode()).hexdigest()

        timestamp = datetime.utcnow().strftime("%Y-%m-%d")
        config = "test" if test else "train"

        identifier = f"{identifier}_{test}_{timestamp}"
        return identifier

    def indies_to_recidx(self, indices):
        rec_indices = []
        k_list = []
        for idx in indices:
            k = 0
            for rec_len in self.seq_lens:
                if idx - rec_len >= 0:
                    idx = idx - rec_len
                    k += 1
                else:
                    rec_indices.append(idx)
                    k_list.append(k)
                    break
        return k_list, rec_indices

    def __getitem__(self, idx):
        data = []
        labels = []
        indices = self.shuffler[idx * self.batch_size:(idx + 1) * self.batch_size]
        if isinstance(self.labels, list):
            k_list, rec_indices = self.indies_to_recidx(indices)
        else:
            indices = np.random.randint(self.labels.shape[0], size=self.batch_size)
            return self.data[indices, ...], self.labels[indices]

        for i in range(self.batch_size):
            seq_indices = rec_indices[i] + np.arange(self.seq_len) * self.seq_step
            data.append(self.data[k_list[i]][seq_indices])
            if self.seq_len > 1:
                same_labels = np.all(self.labels[k_list[i]][seq_indices] == self.labels[k_list[i]][rec_indices[i]])
            else:
                same_labels = True
            if not same_labels:
                majority_label = np.sum(self.labels[k_list[i]][seq_indices], axis=0) // self.seq_len
                labels.append(majority_label)
            else:
                labels.append(self.labels[k_list[i]][rec_indices[i]])

        if self.reshape:
            new_shape = (16, 10, self.seq_len*8)
            for i, sample in enumerate(data):
                transposed_array = np.transpose(sample, (1, 2, 3, 0))
                data[i] = np.reshape(transposed_array, new_shape)
        
        if np.random.rand() > 0.5:
            return np.flip(np.stack(data, axis=0), axis=-2), np.stack(labels, axis=0)
        else:
            return np.stack(data, axis=0), np.stack(labels, axis=0)

    def on_epoch_end(self):
        np.random.shuffle(self.shuffler)
        
    def load_raw_data(self, parent_path, classification : bool = False, test : bool = False, l_count : int = 4, sequence : int = 1) -> tuple:
        if isinstance(parent_path, list):
            folder_list = []
            for p in parent_path:
                folders = list(map(lambda item: p + item, list(os.listdir(p))))
                folder_list = folder_list + folders
        else:
            folder_list = list(map(lambda item: parent_path + item, list(os.listdir(parent_path))))

        if test:
            folder_list = [f for f in folder_list if ('test' in f)]
        else:
            folder_list = [f for f in folder_list if not ('test' in f)]

        print(folder_list)

        data_list = []
        label_list = []

        count = 0
        bar_len = 20

        for folder in folder_list:
            rec_list = os.listdir(folder)  # List of filenames of files inside directory
    #         print(folder)
            label_rec_list = [r for r in rec_list if r.__contains__("label")]
            rec_list = [r for r in rec_list if
                        r.__contains__("radar") and not r.__contains__("label") and not r.__contains__("process")]
            num_files = len(rec_list)
            for i, rec in enumerate(rec_list):
                # print("rec: ", folder + '/' + rec)

                clear_output(wait=True)
                count += 1
                print("Directory: " + parent_path)
                print("Current file: " + rec)
                print("Progress: " + str(count) + "/" + str(num_files))

                data = np.load(folder + '/' + rec)
                # print("Data shape: " + data.shape)
                # print(f"raw data min: {np.amin(data)}, max {np.amax(data)} mean {np.mean(data)}")
                data_processed = self.preprocess_func(data / 4095.0, normalization=False)
                if "all_recs" in folder + rec:
                    #print("got labels :)")
                    label_rec_path = [l for l in label_rec_list if l.__contains__(rec[:-9])]
                    # print(rec, label_rec_path)
                    try:
                        labels = np.load(folder + "/" + label_rec_path[0])
                        labels = np.where(labels > 4, 4, labels)

                    except Exception as e:
                        print(e)
                        print(data.shape)
                        labels = np.load(folder + "/" + label_rec_path[0], allow_pickle=True)

                else:
                    count = int(rec[0])
                    labels = np.ones(data_processed.shape[0]) * count
                    labels = np.where(labels > 4, 4, labels)

                if classification:
                    labels[labels > (l_count - 1)] = l_count - 1
                    b = np.zeros((labels.size, l_count))
                    b[np.arange(labels.size), labels.astype(np.int)] = 1
                    labels = b
                # print(labels.shape[0], data_processed.shape[0])
                # print(f"label available {np.unique(labels)}")
                assert labels.shape[0] == data_processed.shape[0]
                ## Changes can be made here -Z
                label_list.append(labels)
                data_list.append(data_processed)
                assert len(label_list) == len(data_list)

        if sequence > 1:
            return data_list, label_list

        return np.concatenate(data_list, axis=0), np.concatenate(label_list, axis=0)
