import torch

# Based on https://github.com/kijai/ComfyUI-KJNodes/blob/8c590fd5a023ee14b5617347567752bf62ea4cd6/nodes/model_optimization_nodes.py#L359


class TorchCompileLoadControlNet:
    CATEGORY = "torch-compile"
    RETURN_TYPES = ("CONTROL_NET",)
    FUNCTION = "compile"

    def __init__(self):
        self._compiled = False

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "controlnet": ("CONTROL_NET",),
                "backend": (["inductor", "cudagraphs"],),
                "fullgraph": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Enable full graph mode"},
                ),
                "mode": (
                    [
                        "default",
                        "max-autotune",
                        "max-autotune-no-cudagraphs",
                        "reduce-overhead",
                    ],
                    {"default": "default"},
                ),
            }
        }

    def compile(self, controlnet, backend, mode, fullgraph):
        if not self._compiled:
            try:
                controlnet.control_model = torch.compile(controlnet.control_model, mode=mode, fullgraph=fullgraph, backend=backend)
                self._compiled = True
            except:
                self._compiled = False
                raise RuntimeError("Failed to compile model")
       
        return (controlnet, )
