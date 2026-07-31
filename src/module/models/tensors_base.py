from abc import abstractmethod
from collections.abc import Sequence

import torch
from torch import Tensor
from torch.types import Device, Number


class ModelTensorsBase:
    """Base class for creating and storing model tensors."""

    def create_io_tensors(
        self,
        input_names: list[str],
        input_types: list[torch.dtype] | torch.dtype,
        input_fill_values: Sequence[Number],
        input_shapes: list[list[int]],
        input_devices: list[Device] | Device,
        output_names: list[str],
        output_types: list[torch.dtype] | torch.dtype,
        output_fill_values: Sequence[Number],
        output_shapes: list[list[int]],
        output_devices: list[Device] | Device,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Create input/output tensors for a model.

        Parameters
        ----------
        input_names : list[str]
            The names the arguments in the model's forward pass.

            For instance, consider the following model's forward pass:
            ```
            def forward(self, img0, img1, timestep, f0): ...
            ```
            Here, `input_names` would be: `["img0", "img1", "timestep", "f0"]`.
        input_types : list[torch.dtype] | torch.dtype
            The data types of the tensors defined by `input_names`.
        input_fill_values : list[torch.Number]
            The values to fill each input tensor with.
        input_shapes : list[list[int]]
            The shapes of the tensors defined by `input_names`.
        input_devices : list[Device] | Device
            The devices where tensors defined by `input_names` will be placed.
        output_names : list[str]
            The names of the returned values in the model's forward pass.

            For instance, consider the following return values:
            ```
            return (warped_img0 * mask + warped_img1 * (1 - mask))[
                :, :, : self.height, : self.width
            ], f1
            ```
            Here, `output_names` would be: `["output", "f1"]`.
        output_types : list[torch.dtype] | torch.dtype
            The data types of the tensors defined by `output_names`.
        output_fill_values : list[torch.Number]
            The values to fill each output tensor with.
        output_shapes : list[list[int]]
            The shapes of the tensors defined by `output_names`.
        output_devices : list[Device] | Device
            The devices where tensors defined by `output_names` will be placed.

        Returns
        -------
        tuple[dict[str, Tensor], dict[str, Tensor]]
            Returns a tuple, where:
                - tuple[0] are the input tensors.
                - tuple[1] are the output tensors.
        """
        if not isinstance(input_types, list):
            input_types = [input_types for _ in range(len(input_names))]
        if not isinstance(input_devices, list):
            input_devices = [input_devices for _ in range(len(input_names))]
        if not isinstance(output_types, list):
            output_types = [output_types for _ in range(len(output_names))]
        if not isinstance(output_devices, list):
            output_devices = [output_devices for _ in range(len(output_names))]

        input_tensors: dict[str, Tensor] = {}
        for iname, itype, ivalue, ishape, idevice in zip(
            input_names,
            input_types,
            input_fill_values,
            input_shapes,
            input_devices,
            strict=True,
        ):
            input_tensors[iname] = torch.full(
                ishape, fill_value=ivalue, dtype=itype, device=idevice
            ).contiguous()

        output_tensors: dict[str, Tensor] = {}
        for oname, otype, ovalue, oshape, odevice in zip(
            output_names,
            output_types,
            output_fill_values,
            output_shapes,
            output_devices,
            strict=True,
        ):
            output_tensors[oname] = torch.full(
                oshape, fill_value=ovalue, dtype=otype, device=odevice
            ).contiguous()

        return (input_tensors, output_tensors)

    @abstractmethod
    def get_dynamic_axes(self) -> dict[str, dict[str, str]]:
        """Returns the dynamic axes of the tensors.

        A dynamic axis is the part of a tensor which may change at runtime.
        For instance, the width and height of a video frame changes depending on the video.

        Example
        -------
        Here's an example of dynamic axes.

        ```py
        {
            "img0": {"2": "height", "3": "width"},
            "img1": {"2": "height", "3": "width"},
            "timestep": {"2": "height", "3": "width"},
            "f0": {"2": "height", "3": "width"},
            "output": {"1": "height", "2": "width"},
        }
        ```

        Returns
        -------
        dict[str, dict[str, str]]
            The key is the name of the input or output and the value is a dictionary
            that contains the dynamic axes of the input or output.
            The key of the value dictionary is the index of the dynamic axis and the
            value is the name of the dynamic axis.
        """
        ...
