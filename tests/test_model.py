import torch

from src.ekg_digitizer.models.model import EnhancedEKGNet


class ImageProperties:
    img_channels = 3  # 3 image channels = RGB image


class SignalProperties:
    base_fs = 500  # 500 signal samples per sec
    lead_ii_duration = 10.0
    other_leads_duration = 2.5


class TestConfig:
    image_properties = ImageProperties()
    signal_properties = SignalProperties()


def test_model_forward_pass():
    config = TestConfig()  # create config object
    model = EnhancedEKGNet(config)  # create NN
    model.eval()  # use the model in test mode

    x = torch.randn(1, 3, 128, 128)  # create fake image (one 128x128 RGB image)

    with torch.no_grad():  # don't calculate gradients
        outputs = model(x)  # the forward pass: run the image through the model
        # outputs is a dictionary containing the "predictions" for each lead
        # each prediction is the model’s reconstructed 1D EKG signal for that lead
        # reconstructed 1D EKG signal looks like a normal ECG waveform plotted over time: [0.01, 0.03, 0.08, 0.02, -0.05, -0.10, 0.20, 0.85, 0.15, -0.08, ...]

    assert len(outputs) == 12  # check that model returns 12 leads

    for lead, output in outputs.items():
        if lead == "II":
            assert output.shape == (1, 5000)  # check that the shape of lead II is correct
        else:
            assert output.shape == (1, 1250)  # check that the shape of the other leads is correct
