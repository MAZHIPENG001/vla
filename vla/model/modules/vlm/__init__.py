


def get_vlm_model(config):
    vlm_name = config.framework.qwenvl.base_vlm

    if "Qwen" in vlm_name:
        from .Qwen import _QWen_VL_Interface

        return _QWen_VL_Interface(config)