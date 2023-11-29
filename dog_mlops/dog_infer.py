import torch
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from utils import predict
from Dog_train import DogDataset


MODEL_DIR = Path('dog_model')
TEST_DIR = Path('data/test')
CSV_DIR = Path('Dogs_predictions.csv')

def infer():
    model = torch.load(MODEL_DIR)
    
    label_encoder = pickle.load(open("label_encoder.pkl", 'rb'))

    test_files = sorted(list(TEST_DIR.rglob('*.jpeg')))
    test_dataset = DogDataset(test_files, mode="test")
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=64)
    probs = predict(model, test_loader)


    preds = label_encoder.inverse_transform(np.argmax(probs, axis=1))
    test_filenames = [path.name for path in test_dataset.files]

    df = pd.DataFrame()
    df['Name'] = test_filenames
    df['Expected'] = preds
    df.to_csv(CSV_DIR, index=False)
    df.head()

if __name__ == "__main__":
    infer()
    print("csv created")