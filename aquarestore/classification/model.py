"""Aquatic Image Classifier model architecture based on MobileNetV3."""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import MobileNet_V3_Small_Weights


class AquaticClassifier(nn.Module):
    """Deep learning binary classifier to identify aquatic and water-containing images."""

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.backbone = models.mobilenet_v3_small(weights=weights)

        # Replace the final linear projection layer with our binary classification head
        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[2] = nn.Dropout(p=dropout, inplace=True)
        self.backbone.classifier[3] = nn.Linear(in_features, num_classes)

    def freeze_backbone(self, freeze: bool = True) -> None:
        """Freeze or unfreeze all feature extraction layers in the backbone."""
        for param in self.backbone.features.parameters():
            param.requires_grad = not freeze

    def unfreeze_top_blocks(self, num_blocks: int = 3) -> None:
        """Unfreeze the top N residual blocks of the backbone for fine-tuning."""
        # Unfreeze classifier head first
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True

        total_blocks = len(self.backbone.features)
        start_idx = max(0, total_blocks - num_blocks)
        for i in range(start_idx, total_blocks):
            for param in self.backbone.features[i].parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits."""
        return self.backbone(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning softmax probability distributions."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)


def build_model(
    num_classes: int = 2,
    pretrained: bool = True,
    device: torch.device | None = None,
) -> AquaticClassifier:
    """Factory function to build and configure the AquaticClassifier."""
    model = AquaticClassifier(num_classes=num_classes, pretrained=pretrained)
    if device is not None:
        model = model.to(device)
    return model

