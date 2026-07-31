import torch

from src.module.models.model_handler import ModelHandler


def load_vfi_model(model_path: str):
    handler = ModelHandler()
    model_params = handler.get_model_parameters(model_path)["vfi"]
    tensor_params = handler.get_tensor_parameters(model_path)["vfi"]
    model_name = handler.get_model_name(model_path)
    match model_name:
        # TODO: Imports can be done programmatically with importlib
        case "rife_elexor":
            from src.module.models.vfi.rife.IFNet_elexor import IFNet

            model = IFNet(**model_params)
            model.load_state_dict(torch.load(model_path))
            model.eval()
            tensor_params["tensors"].ENCODE = model.encode
            return model
        case "rife425":
            from src.module.models.vfi.rife.Rife425_v3 import IFNet

            model = IFNet(**model_params)
            model.load_state_dict(torch.load(model_path))
            model.eval()
            tensor_params["tensors"].ENCODE = model.encode
            return model


def test_saver(model_path: str):
    import os
    from pathlib import Path

    import onnxruntime
    from onnx import TensorProto

    from src.module.config.onnx_config.run_option_config import RunOptionsConfig
    from src.module.models.model_base import ModelBase

    inference_tensor_device = "cuda"
    model_params = {
        "width": 1920,
        "height": 1080,
        "padded_width": 1920,
        "padded_height": 1088,
        "dtype": torch.float32,
        "scale": 1,
        "device": "cuda",
    }
    frame_shape = [1, 3, 1080, 1920]
    padded_frame_shape = [1, 3, 1088, 1920]
    timestep_shape = [1, 1, 1088, 1920]
    encoded_frame_shape = [1, 4, 1088, 1920]
    input_names = ["img0", "img1", "timestep", "f0"]
    output_names = ["prediction", "f1"]

    if not os.path.splitext(os.path.split(model_path)[1])[1] == ".onnx":
        from src.module.models.vfi.rife.IFNet_elexor import IFNet

        model = IFNet(**model_params)
        model.load_state_dict(torch.load(model_path))
        model.to(dtype=model_params["dtype"], device=model_params["device"])
        model.eval()

        I0_IN = torch.zeros(
            padded_frame_shape,
            dtype=model_params["dtype"],
            device=model_params["device"],
        ).contiguous()
        I1_IN = torch.zeros(
            padded_frame_shape,
            dtype=model_params["dtype"],
            device=model_params["device"],
        ).contiguous()
        TIMESTEP_IN = torch.full(
            timestep_shape,
            0.5,
            dtype=model_params["dtype"],
            device=model_params["device"],
        ).contiguous()
        F0_IN = torch.zeros(
            encoded_frame_shape,
            dtype=model_params["dtype"],
            device=model_params["device"],
        ).contiguous()

        program = torch.onnx.export(
            model,
            args=(I0_IN, I1_IN, TIMESTEP_IN, F0_IN),
            input_names=input_names,
            output_names=output_names,
            # dynamic_axes={
            #     input_names[0]: {2: "height", 3: "width"},
            #     input_names[1]: {2: "height", 3: "width"},
            #     input_names[2]: {2: "height", 3: "width"},
            #     input_names[3]: {2: "height", 3: "width"},
            #     output_names[0]: {1: "height", 2: "width"},
            # },
            # dynamic_shapes={
            #     input_names[0]: {2: Dim.DYNAMIC, 3: Dim.DYNAMIC},
            #     input_names[1]: {2: Dim.DYNAMIC, 3: Dim.DYNAMIC},
            #     input_names[2]: {2: Dim.DYNAMIC, 3: Dim.DYNAMIC},
            #     input_names[3]: {2: Dim.DYNAMIC, 3: Dim.DYNAMIC},
            # },
            dynamo=True,
            optimize=True,
            verify=True,
            export_params=True,
        )

        out_path = f"{Path('/home/cachy/Programming_Projects/TheAnimeScripter/weights/vfi/rife_elexor.onnx')}"
        program.save(  # type: ignore
            out_path
        )
    else:
        out_path = model_path

    session_options = ModelBase.get_session_configuration()
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    provider_options = ModelBase.get_provider_configs(providers)

    session = onnxruntime.InferenceSession(
        out_path,
        sess_options=session_options,
        providers=providers,
        provider_options=provider_options,
    )

    I0_IN = torch.zeros(
        padded_frame_shape, dtype=model_params["dtype"], device=inference_tensor_device
    ).contiguous()
    I1_IN = torch.zeros(
        padded_frame_shape, dtype=model_params["dtype"], device=inference_tensor_device
    ).contiguous()
    TIMESTEP_IN = torch.full(
        timestep_shape, 0.5, dtype=model_params["dtype"], device=inference_tensor_device
    ).contiguous()
    F0_IN = torch.zeros(
        encoded_frame_shape, dtype=model_params["dtype"], device=inference_tensor_device
    ).contiguous()
    F1_OUT = torch.zeros(
        encoded_frame_shape, dtype=model_params["dtype"], device=inference_tensor_device
    ).contiguous()
    PREDICTION_OUT = torch.zeros(
        frame_shape, dtype=model_params["dtype"], device=inference_tensor_device
    ).contiguous()

    io_binding = session.io_binding()
    # OnnxRuntime will copy the data over to the CUDA device if 'input' is consumed by nodes on the CUDA device
    io_binding.bind_input(
        name=input_names[0],
        device_type=I0_IN.device.type,
        device_id=0 if I0_IN.device.index is None else I0_IN.device.index,
        element_type=TensorProto.FLOAT,
        shape=tuple(I0_IN.shape),
        buffer_ptr=I0_IN.data_ptr(),
    )
    io_binding.bind_input(
        name=input_names[1],
        device_type=I1_IN.device.type,
        device_id=0 if I1_IN.device.index is None else I1_IN.device.index,
        element_type=TensorProto.FLOAT,
        shape=tuple(I1_IN.shape),
        buffer_ptr=I1_IN.data_ptr(),
    )
    io_binding.bind_input(
        name=input_names[2],
        device_type=TIMESTEP_IN.device.type,
        device_id=0 if TIMESTEP_IN.device.index is None else TIMESTEP_IN.device.index,
        element_type=TensorProto.FLOAT,
        shape=tuple(TIMESTEP_IN.shape),
        buffer_ptr=TIMESTEP_IN.data_ptr(),
    )
    io_binding.bind_input(
        name=input_names[3],
        device_type=F0_IN.device.type,
        device_id=0 if F0_IN.device.index is None else F0_IN.device.index,
        element_type=TensorProto.FLOAT,
        shape=tuple(F0_IN.shape),
        buffer_ptr=F0_IN.data_ptr(),
    )
    io_binding.bind_output(
        name=output_names[0],
        device_type=PREDICTION_OUT.device.type,
        device_id=0
        if PREDICTION_OUT.device.index is None
        else PREDICTION_OUT.device.index,
        element_type=TensorProto.FLOAT,
        shape=tuple(PREDICTION_OUT.shape),
        buffer_ptr=PREDICTION_OUT.data_ptr(),
    )
    io_binding.bind_output(
        name=output_names[1],
        device_type=F1_OUT.device.type,
        device_id=0 if F1_OUT.device.index is None else F1_OUT.device.index,
        element_type=TensorProto.FLOAT,
        shape=tuple(F1_OUT.shape),
        buffer_ptr=F1_OUT.data_ptr(),
    )

    run_option_config = RunOptionsConfig()
    run_options = onnxruntime.RunOptions()
    for k, v, _ in run_option_config:
        run_options.add_run_config_entry(k, v)

    print("running io session")
    for _ in range(6):
        I0_IN.copy_(torch.full(padded_frame_shape, 7))

        session.run_with_iobinding(io_binding, run_options=run_options)

        print(PREDICTION_OUT)
        print(F1_OUT)

        I0_IN.copy_(torch.full(padded_frame_shape, 4))
        I1_IN.fill_(3)
        F0_IN.fill_(2)

    # print("running session")
    # output = session.run(
    #     output_names,
    #     {
    #         input_names[0]: I0_IN.cpu().numpy(),
    #         input_names[1]: I1_IN.cpu().numpy(),
    #         input_names[2]: TIMESTEP_IN.cpu().numpy(),
    #         input_names[3]: F0_IN.cpu().numpy(),
    #     },
    # )
    # print(output)


if __name__ == "__main__":
    test_saver(
        "/home/cachy/Programming_Projects/TheAnimeScripter/weights/vfi/rife_elexor.onnx"
    )
