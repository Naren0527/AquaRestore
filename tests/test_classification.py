"""Unit tests for the deep learning classifier module."""

import numpy as np
import pytest
import torch
from PIL import Image

from aquarestore.classification.config import ClassifierConfig
from aquarestore.classification.dataset import (
    AquaticDataset,
    get_eval_transforms,
    get_train_transforms,
)
from aquarestore.classification.inference import (
    AquaticInference,
    ClassificationResult,
)
from aquarestore.classification.model import AquaticClassifier, build_model


def test_classifier_architecture_and_forward_pass():
    model = AquaticClassifier(num_classes=2, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)

    assert logits.shape == (2, 2)
    assert not torch.isnan(logits).any()


def test_classifier_predict_proba():
    model = AquaticClassifier(num_classes=2, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    probs = model.predict_proba(x)

    assert probs.shape == (2, 2)
    assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_backbone_freezing_and_unfreezing():
    model = AquaticClassifier(num_classes=2, pretrained=False)

    model.freeze_backbone(freeze=True)
    for param in model.backbone.features.parameters():
        assert not param.requires_grad

    # Classifier head should remain trainable
    for param in model.backbone.classifier.parameters():
        assert param.requires_grad

    model.unfreeze_top_blocks(num_blocks=2)
    # Check that the last feature block has trainable parameters
    last_block = model.backbone.features[-1]
    assert any(p.requires_grad for p in last_block.parameters())


def test_dataset_transforms():
    config = ClassifierConfig()
    train_tf = get_train_transforms(config)
    eval_tf = get_eval_transforms(config)

    dummy_pil = Image.fromarray(np.zeros((300, 300, 3), dtype=np.uint8))
    tensor_train = train_tf(dummy_pil)
    tensor_eval = eval_tf(dummy_pil)

    assert tensor_train.shape == (3, 224, 224)
    assert tensor_eval.shape == (3, 224, 224)


def test_inference_with_mocked_model(monkeypatch):
    class MockModel(torch.nn.Module):
        def __init__(self, prob_aq):
            super().__init__()
            self.prob_aq = prob_aq

        def predict_proba(self, x):
            return torch.tensor([[1.0 - self.prob_aq, self.prob_aq]])

    engine = AquaticInference(model_path="non_existent.pth", threshold=0.85)

    # 1. High confidence aquatic
    engine.model = MockModel(prob_aq=0.96)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    res = engine.predict(img)
    assert res.is_aquatic is True
    assert res.accepted is True
    assert res.confidence == pytest.approx(0.96, abs=1e-3)
    assert "Aquatic Image Detected" in res.message

    # 2. Low confidence aquatic (below threshold)
    engine.model = MockModel(prob_aq=0.72)
    res = engine.predict(img)
    assert res.is_aquatic is True
    assert res.accepted is False
    assert "Uncertain Image Content" in res.message

    # 3. Non-aquatic
    engine.model = MockModel(prob_aq=0.05)
    res = engine.predict(img)
    assert res.is_aquatic is False
    assert res.accepted is False
    assert "Unsupported Image" in res.message

