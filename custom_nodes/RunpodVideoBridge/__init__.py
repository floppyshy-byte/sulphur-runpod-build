import os


class RunpodVideoBridge:
    """Bridge VHS video outputs to standard ComfyUI image outputs for RunPod handler."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "filenames": ("VHS_FILENAMES",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "bridge_output"
    OUTPUT_NODE = True
    CATEGORY = "Utility/Bridges"

    def bridge_output(self, images, filenames):
        result = []

        # Unpack the VideoHelperSuite tuple wrapper safely
        actual_files = []
        if isinstance(filenames, (tuple, list)):
            for item in filenames:
                if isinstance(item, list):
                    actual_files = item
                    break
        elif isinstance(filenames, list):
            actual_files = filenames

        # Process the unpacked physical file names
        if actual_files:
            for file_path in actual_files:
                if isinstance(file_path, str):
                    base_name = os.path.basename(file_path)
                    result.append({
                        "filename": base_name,
                        "subfolder": "",
                        "type": "output"
                    })

        # Inject into ComfyUI's UI output map so RunPod handler picks it up
        return {"ui": {"images": result}, "result": (images,)}


NODE_CLASS_MAPPINGS = {
    "RunpodVideoBridge": RunpodVideoBridge,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RunpodVideoBridge": "Runpod Video Output Bridge",
}
