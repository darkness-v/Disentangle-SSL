import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import os
import librosa
class ESDDataset(Dataset):
    def __init__(self, data_dir, cut = 66800):
        self.data_dir = data_dir
        self.speakers = ["0011", "0012", "0013", "0014", "0015", "0016", "0017", "0018", "0019", "0020"]
        self.emotions = ["Neutral", "Angry", "Happy", "Sad", "Surprise"]
        self.file_list = []
        self.cut = cut
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
    def padding(self, waveform, target_length):
        if waveform.size(0) > target_length:
            return waveform[:target_length]
        num_repeats = int(target_length / waveform.size(0)) + 1
        padded_waveform = waveform.repeat(num_repeats)[:target_length]
        return padded_waveform
    def __getitem__(self, idx):
        files = self.data[idx]
        waveforms = []
        for file in files:
            waveform, _ = librosa.load(file, sr=16000)
            waveform = torch.tensor(waveform)
            waveform = self.padding(waveform, self.cut)
            waveforms.append(waveform)
        return torch.stack([torch.tensor(wf) for wf in waveforms], dim=0)  # Shape: [5, T]

if __name__ == "__main__":
    dataset = ESDDataset(data_dir='data/Emotion Speech Dataset')
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    for batch in dataloader:
        print(f"Batch size: {len(batch)}")
        for sample in batch:
            print(sample.shape)
            sample = torch.tensor(sample)
            print(f"Sample shape: {sample.shape}")  # Should be [5, T
        break