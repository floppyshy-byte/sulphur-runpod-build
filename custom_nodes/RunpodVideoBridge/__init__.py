import os


class RunpodVideoBridge:
    """Bridge VHS video outputs to standard ComfyUI image outputs for RunPod handler.
    Also captures enhanced prompt text and writes it to the output directory so
    RunPod's file scanner uploads it alongside the video."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "filenames": ("VHS_FILENAMES",),
            },
            "optional": {
                "text": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "bridge_output"
    OUTPUT_NODE = True
    CATEGORY = "Utility/Bridges"

    def bridge_output(self, images, filenames, text=None):
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

        # Write enhanced prompt text to output directory so RunPod uploads it
        if text and isinstance(text, str) and text.strip():
            output_dir = "/comfyui/output"
            os.makedirs(output_dir, exist_ok=True)
            txt_path = os.path.join(output_dir, "enhanced_prompt.txt")
            try:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(text)
                result.append({
                    "filename": "enhanced_prompt.txt",
                    "subfolder": "",
                    "type": "output"
                })
            except Exception:
                pass

        # Inject into ComfyUI's UI output map so RunPod handler picks it up
        return {"ui": {"images": result}, "result": (images,)}


class StripThinkingTags:
    """Strips thinking/reasoning tags from prompt enhancer output before
    feeding to CLIPTextEncode. Keeps the upstream custom node unpatched."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "strip"
    OUTPUT_NODE = False
    CATEGORY = "Utility/Text"

    def strip(self, text):
        import re
        cleaned = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL).strip()
        return (cleaned,)


NODE_CLASS_MAPPINGS = {
    "RunpodVideoBridge": RunpodVideoBridge,
    "StripThinkingTags": StripThinkingTags,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RunpodVideoBridge": "Runpod Video Output Bridge",
    "StripThinkingTags": "Strip Thinking Tags",
}
