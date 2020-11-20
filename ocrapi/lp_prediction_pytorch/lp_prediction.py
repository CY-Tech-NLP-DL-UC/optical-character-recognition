import io
from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image

from .models import CRNN

model_folder = Path(__file__).resolve().parent / "models"


class Decoder(object):
    def __init__(self, characters):
        self.characters = characters

    def __call__(self, inputs) -> str:
        arr = inputs.numpy()
        decoded = [self.characters[idx] for idx in arr]
        output = "".join(
            [
                decoded[i]
                for i in range(len(decoded) - 1)
                if decoded[i] != decoded[i + 1]
            ]
            + [decoded[len(decoded) - 1]]
        )
        output = "".join([c for c in output if c != self.characters[0]])
        return output


def transform_images(image_bytes):
    my_transforms = transforms.Compose(
        [transforms.Resize((32, 128)), transforms.ToTensor()]
    )

    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
    img_height = 32
    channel_number = 1
    alphabet = [
        "_",
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "J",
        "K",
        "L",
        "M",
        "N",
        "O",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "U",
        "V",
        "W",
        "X",
        "Y",
        "Z",
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
    ]

    number_hidden_layers = 256

    model = CRNN(
        img_height, channel_number, len(alphabet), number_hidden_layers
    )
    model.load_state_dict(torch.load(model_folder / "crnn.pth"))
    model.eval()

    tensor = transform_images(image_bytes=image_bytes)
    output = model.forward(tensor)
    _, pred = output.max(2)
    pred = pred.transpose(1, 0).contiguous().squeeze(0)
    decoder = Decoder(alphabet)
    pred = decoder(pred)
    return pred
