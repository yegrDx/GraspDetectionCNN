from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class GGCNN(nn.Module):

    def __init__(self, in_ch: int = 1):
        super().__init__()
        if in_ch != 1:
            raise ValueError("GGCNN canonical version expects 1-channel depth input.")
        self.conv1 = nn.Conv2d(1, 32, kernel_size=9, stride=3, padding=3)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=1)

        self.convt1 = nn.ConvTranspose2d(8, 8, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.convt2 = nn.ConvTranspose2d(8, 16, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.convt3 = nn.ConvTranspose2d(16, 32, kernel_size=9, stride=3, padding=3, output_padding=1)

        self.pos_output = nn.Conv2d(32, 1, kernel_size=2, stride=1)
        self.cos_output = nn.Conv2d(32, 1, kernel_size=2, stride=1)
        self.sin_output = nn.Conv2d(32, 1, kernel_size=2, stride=1)
        self.width_output = nn.Conv2d(32, 1, kernel_size=2, stride=1)

        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = F.relu(self.convt1(x))
        x = F.relu(self.convt2(x))
        x = F.relu(self.convt3(x))

        pos = torch.sigmoid(self.pos_output(x))
        cos = torch.tanh(self.cos_output(x))
        sin = torch.tanh(self.sin_output(x))
        width = torch.sigmoid(self.width_output(x))
        return pos, cos, sin, width


class GGCNN2(nn.Module):
  
    def __init__(self, in_ch: int = 1, base_ch: int = 32, dropout: float = 0.0):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 9, stride=3, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch // 2, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch // 2, base_ch // 4, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(base_ch // 4, base_ch // 4, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_ch // 4, base_ch // 2, 5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_ch // 2, base_ch, 9, stride=3, padding=3, output_padding=1),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        self.pos_output = nn.Conv2d(base_ch, 1, 2)
        self.cos_output = nn.Conv2d(base_ch, 1, 2)
        self.sin_output = nn.Conv2d(base_ch, 1, 2)
        self.width_output = nn.Conv2d(base_ch, 1, 2)

        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        x = self.enc(x)
        x = self.dec(x)
        x = self.dropout(x)
        pos = torch.sigmoid(self.pos_output(x))
        cos = torch.tanh(self.cos_output(x))
        sin = torch.tanh(self.sin_output(x))
        width = torch.sigmoid(self.width_output(x))
        return pos, cos, sin, width


def build_model(name: str, in_ch: int = 1) -> nn.Module:
    name = name.lower()
    if name == "ggcnn":
        return GGCNN(in_ch=in_ch)
    if name == "ggcnn2":
        return GGCNN2(in_ch=in_ch)
    
