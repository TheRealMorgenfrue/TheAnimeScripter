from applib import ComboBoxOption, GUIMessage, TextEditOption


def tas_encoding_template() -> dict:
    return {
        "encode_method": ComboBoxOption(
            default="x264",
            values=[
                "x264",
                "slow_x264",
                "x264_10bit",
                "x264_animation",
                "x264_animation_10bit",
                "x265",
                "slow_x265",
                "x265_10bit",
                "nvenc_h264",
                "slow_nvenc_h264",
                "nvenc_h265",
                "slow_nvenc_h265",
                "nvenc_h265_10bit",
                "nvenc_av1",
                "slow_nvenc_av1",
                "qsv_h264",
                "qsv_h265",
                "qsv_h265_10bit",
                "av1",
                "slow_av1",
                "h264_amf",
                "hevc_amf",
                "hevc_amf_10bit",
                "prores",
                "prores_segment",
                "gif",
                "vp9",
                "qsv_vp9",
                "lossless",
                "lossless_nvenc",
                "png",
                "nvenc_h264_nelux",
                "nvenc_h265_nelux",
                "nvenc_av1_nelux",
            ],
            ui_info=GUIMessage("Encoding method"),
        ),
        "custom_encoder": TextEditOption(
            default="", ui_info=GUIMessage("Custom encoder settings")
        ),
        "bit_depth": ComboBoxOption(
            default="8bit",
            values=["8bit", "16bit"],
            ui_info=GUIMessage("Bit Depth of the raw pipe input to FFmpeg"),
        ),
    }
