import torchvision.transforms as transforms
from Vocabulary import Vocabulary
from DndDataset import myData, MyCollate
from torch.utils.data import DataLoader

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.ColorJitter(
        brightness=0.2,  # Variazione luminosità (±20%)
        contrast=0.2,    # Variazione contrasto (±20%)
        saturation=0.2,  # Variazione saturazione (±20%)
        hue=0.1          # Variazione tonalità (±10%)
    ),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transorm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

vocabulary = Vocabulary()
vocabulary.build_vocabulary('./monsters_dataset.csv')

train_data = myData(csv_path='./monsters_train.csv', transform=train_transform, vocabulary=vocabulary)
val_data = myData(csv_path='./monsters_val.csv', transform=test_transorm, vocabulary=vocabulary)
test_data = myData(csv_path='./monsters_test.csv', transform=test_transorm, vocabulary=vocabulary)

collate = MyCollate()
train_loader = DataLoader(train_data, 8, shuffle=True, collate_fn=collate)
val_loader = DataLoader(val_data, 8, shuffle=False, collate_fn=collate)
test_loader = DataLoader(test_data, 8, shuffle=False, collate_fn=collate)