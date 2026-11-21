from ekg_digitizer.config import HardwareConfig, load_config


def test_hardware_config_initialization():
    config = HardwareConfig()

    assert isinstance(config.use_multi_gpu, bool)
    assert isinstance(config.device_ids, list)
    assert len(config.device_ids) >= 1


def test_config_initialization():
    config = load_config("configs/config.yaml")

    assert config.hardware is not None
    assert isinstance(config.hardware.use_multi_gpu, bool)
    assert isinstance(config.hardware.device_ids, list)
