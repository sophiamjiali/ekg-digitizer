import torch

from src.ekg_digitizer.models.model import EnhancedEKGNet


class TestConfig:
    img_channels = 3
    base_fs = 500
    lead_ii_duration = 10.0
    other_leads_duration = 2.5


def test_model_forward_pass():
    config = TestConfig()
    model = EnhancedEKGNet(config)
    model.eval()

    x = torch.randn(1, 3, 128, 128)

    with torch.no_grad():
        outputs = model(x)

    assert len(outputs) == 12

    for lead, output in outputs.items():
        if lead == "II":
            assert output.shape == (1, 5000)
        else:
            assert output.shape == (1, 1250)
