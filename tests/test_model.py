import torch

from ekg_digitizer.models.model import AdvancedTCN, EnhancedEKGNet


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


def test_model_preserves_horizontal_sequence_for_decoders():
    model = EnhancedEKGNet(TestConfig()).eval()
    captured = {}

    def capture_fusion(_module, _inputs, output):
        captured["fusion"] = output.shape

    def capture_decoder_input(_module, inputs):
        captured["decoder_input"] = inputs[0].shape

    fusion_hook = model.fusion.register_forward_hook(capture_fusion)
    decoder_hook = model.lead_decoders["I"].register_forward_pre_hook(capture_decoder_input)

    try:
        with torch.no_grad():
            model(torch.randn(1, 3, 128, 160))
    finally:
        fusion_hook.remove()
        decoder_hook.remove()

    assert captured["fusion"][3] > 1
    assert captured["decoder_input"] == (
        captured["fusion"][0],
        captured["fusion"][1],
        captured["fusion"][3],
    )
    assert captured["decoder_input"][2] > 1


def test_advanced_tcn_accepts_horizontal_sequence():
    decoder = AdvancedTCN(input_dim=8, output_length=40, num_channels=(8, 8)).eval()

    with torch.no_grad():
        output = decoder(torch.randn(2, 8, 5))

    assert output.shape == (2, 40)
