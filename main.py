import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'bi-lstm-crf')))

from bi_lstm_crf.app import WordsTagger

model = WordsTagger(model_dir="D:/CS221-NaturalLanguageProcessing/FinalProject/NamedEntityRecognition/saved_model")
tags, sequences = model([["cái", "điều", "thứ", "hai", "đó", "là", "đối", "với", "những", "các", "thành", "phần", "kem", "dưỡng", "ẩm", "vào", "ban", "đêm", "thường", "sẽ", "có", "cái", "nồng", "độ", "của", "những", "hoạt"]])  # CHAR-based model
print(tags)  
print(sequences)

import pandas as pd
import matplotlib.pyplot as plt

# the training losses are saved in the model_dir
df = pd.read_csv("saved_model/loss.csv")
df[["train_loss", "val_loss"]].ffill().plot(grid=True)
plt.show()