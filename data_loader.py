import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import os
import librosa
class ESDDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.speakers = ["0011", "0012", "0013", "0014", "0015", "0016", "0017", "0018", "0019", "0020"]
        self.emotions = ["Neutral", "Angry", "Happy", "Sad", "Surprise"]
        self.file_list = []
        for speaker in self.speakers:
            for emotion in self.emotions:
                emotion_dir = os.path.join(self.data_dir, speaker, emotion)
                for file_name in os.listdir(emotion_dir):
                    if file_name.endswith('.wav'):
                        self.file_list.append(os.path.join(emotion_dir, file_name))
        self.data = [self.file_list[i:i+5] for i in range(0, len(self.file_list), 5)]

        print(f"Total samples in dataset: {len(self.data)}")
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        files = self.data[idx]
        waveforms = []
        for file in files:
            waveform, _ = librosa.load(file, sr=16000)
            waveforms.append(waveform)
        return waveforms

# if __name__ == "__main__":
#     dataset = ESDDataset(data_dir='data/Emotion Speech Dataset')
#     dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x)
    