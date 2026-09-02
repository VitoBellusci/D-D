from PIL import Image
from torch.utils.data import Dataset
from Vocabulary import Vocabulary
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pad_sequence

class myData(Dataset):
    def __init__(self, csv_path, transform, vocabulary = Vocabulary):
        super().__init__()

        self.df = pd.read_csv(csv_path)
        self.vocabulary = vocabulary
        self.transform = transform

    def __getitem__(self, index):
        row = self.df.iloc[index]
        caption = row['caption']
        image_path = row['image_path']

        image = Image.open(image_path).convert('RGB')
        token = self.vocabulary.numericalize(sentence = caption)

        image_tensor = self.transform(image)
        token_tensor = torch.tensor(token, dtype=torch.long)

        return image_tensor, token_tensor

    def __len__(self):
        return len(self.df)


class MyCollate():
    def __init__(self):
        self.pad_idx = 0
    def __call__(self, batch):
        images = [image[0] for image in batch]
        tensors = [tensor[1] for tensor in batch]

        image_stack = torch.stack(images)
        aligned_tensor = pad_sequence(tensors, batch_first=True, padding_value=self.pad_idx)

        return image_stack, aligned_tensor