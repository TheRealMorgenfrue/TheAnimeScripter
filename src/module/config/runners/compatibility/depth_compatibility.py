def compatible_depth_model(depth_quality: str, depth_method: str):
    if depth_quality != "low" and depth_method.split("-")[-1] in [
        "tensorrt",
        "directml",
    ]:
        raise ValueError(
            f"{depth_quality.title()} depth estimation quality is incomaptible with TensorRT and DirectML"
        )
