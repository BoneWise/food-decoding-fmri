from __future__ import annotations

import torch
from torch import nn


class FMRIConvNet(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, int, int],
        *,
        hidden_dim: int = 128,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.input_shape = tuple(int(dim) for dim in input_shape)
        self.dropout = float(dropout)
        self.feature_extractor = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=7, stride=2, padding=0),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
            nn.Conv3d(8, 16, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=3, stride=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=self.dropout),
        )
        self.flatten_dim = self._infer_flatten_dim()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_dim, int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden_dim), 1),
        )

    def _infer_flatten_dim(self) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 1, *self.input_shape, dtype=torch.float32)
            features = self.feature_extractor(dummy)
        return int(features.reshape(1, -1).shape[1])

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        return features

    def forward_raw(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        logits = self.classifier(features)
        return logits.squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward_raw(x))
