import torch

from fair2026_lms_kan.models.classifier import build_model
from fair2026_lms_kan.models.mobileunet import build_mobileunet


def test_classifier_heads_forward():
    inputs = torch.randn(2, 3, 224, 224)
    for head in ("fastkan", "linear", "mlp"):
        model = build_model(head=head, backbone_size="pico")
        outputs = model(inputs)
        assert outputs.shape == (2, 2)


def test_mobileunet_forward():
    model = build_mobileunet()
    outputs = model(torch.randn(2, 3, 256, 256))
    assert outputs.shape == (2, 1, 256, 256)

