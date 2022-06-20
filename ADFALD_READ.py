import os
import random
import numpy as np
from sklearn.model_selection import train_test_split
from torch import norm

class ADFA_DataLoader () :
    def get_training_data(self):
        train_data_path = "ADFA-LD/Training_Data_Master"
        files = os.listdir(train_data_path)

        list = []
        for i in files :
            f = open(train_data_path + "/" + i, "r")
            list.append(f.read().strip().split(" "))
            f.close()

        return list

    def get_validation_data(self) :
        validation_data_path = "ADFA-LD/Validation_Data_Master"
        files = os.listdir(validation_data_path)

        list = []
        for i in files :
            f = open(validation_data_path + "/" + i, "r")
            list.append(f.read().strip().split(" "))
            f.close()

        return list

    def get_attack_data(self) :
        attack_data_path = "ADFA-LD/Attack_Data_Master"

        folders = os.listdir(attack_data_path)

        list = []
        category = []

        for i in folders :
            files = os.listdir(attack_data_path + "/" + i)
            for j in files :
                f = open(attack_data_path + "/" + i + "/" + j, "r")
                list.append(f.read().strip().split(" "))
                category.append(i)
                f.close()

        return list, category

    def split_data(self):
        normal_data, normal_label, attack_data, attack_label = self.get_all_data()

        x_train, x_test, y_train, y_test = train_test_split(normal_data + attack_data, normal_label + attack_label, test_size = 0.4, random_state=42)
        x_test, x_vali, y_test, y_vali = train_test_split(x_test, y_test, test_size = 0.5, random_state=42)

        return x_train, y_train, x_vali, y_vali, x_test, y_test


    def get_all_data(self):
        normal_data = self.get_training_data() + self.get_validation_data()
        attack_data, category = self.get_attack_data()

        normal_label = [0 for i in range(len(normal_data))]
        attack_label = [0 for i in range(len(attack_data))]

        return normal_data, normal_label, attack_data, attack_label

    def preprocess(self, sentences):

        syscall = []

        for sentence in sentences :
            for token in sentence :
                if token not in syscall:
                    syscall.append(token)

        syscall2idx = {w : idx for (idx, w) in enumerate(syscall)}
        idx2syscall = {idx : w for (idx, w) in enumerate(syscall)}

        corpus = []

        for sentence in sentences :
            corpus.append([syscall2idx[token] for token in sentence])

        return corpus, syscall2idx, idx2syscall

    def create_contexts_target(self, corpus, window_size):
        start_window, end_window = window_size

        target = []
        contexts = []

        for sentence in corpus:
            if len(sentence) <= start_window + end_window:
                continue

            for idx in range(start_window, len(sentence) - end_window):
                target.append(sentence[idx])

                cs = []
                for t in range(-start_window, end_window + 1):
                    if t == 0:
                        continue
                    cs.append(sentence[idx + t])

                contexts.append(cs)

        return np.array(contexts), np.array(target)

    def create_co_matrix(self, corpus, size, window_size):

        start_window, end_window = window_size

        co_matrix = np.zeros((size, size), dtype=np.float32)

        for sentence in corpus :
            for idx, word_id in enumerate(sentence):
                for i in range(1, start_window + 1):
                    left_idx = idx - i
                    if left_idx >= 0:
                        left_word_id = sentence[left_idx]
                        co_matrix[word_id, left_word_id] += 1

                for i in range(1, end_window + 1):
                    right_idx = idx + i
                    if right_idx < len(sentence):
                        right_word_id = sentence[right_idx]
                        co_matrix[word_id, right_word_id] += 1

        return co_matrix

