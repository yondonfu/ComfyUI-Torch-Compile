import torch


class TorchCompileModel:
    CATEGORY = "torch-compile"
    RETURN_TYPES = ("TORCH_MODEL",)
    FUNCTION = "compile"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("TORCH_MODEL",),
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

    def compile(self, model, backend, mode, fullgraph):
        compiled_model = torch.compile(model, mode=mode, fullgraph=fullgraph, backend=backend)
        return (compiled_model,)
