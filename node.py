# Use a pipeline as a high-level helper
import os
from datetime import datetime
from PIL import Image, PngImagePlugin
from transformers import pipeline
import torchvision.transforms as T
import torch
import numpy # Kept for tensor2pil/pil2tensor


def tensor2pil(image): # Remains for potential internal use or debugging, not directly used by new run logic
    return Image.fromarray(numpy.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(numpy.uint8))


def pil2tensor(image): # Remains for potential internal use or debugging
    return torch.from_numpy(numpy.array(image).astype(numpy.float32) / 255.0).unsqueeze(0)


class NSFWDetection:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",), # ComfyUI IMAGE tensor (batch)
                "score": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "round": 0.001,
                    "display": "nsfw_threshold"}),
                "file_paths": ("STRING", {"multiline": True, "default": "", "display": "File Paths (one per line)"}),
                "sfw_output_dir": ("STRING", {"default": "output/sfw", "display": "SFW Output Directory"}),
                "nsfw_output_dir": ("STRING", {"default": "output/nsfw", "display": "NSFW Output Directory"}),
            },
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()

    FUNCTION = "run"

    CATEGORY = "NSFWDetection"

    def run(self, image, score, file_paths, sfw_output_dir, nsfw_output_dir):
        paths_list = [p.strip() for p in file_paths.strip().split('\n') if p.strip()]

        if len(image) != len(paths_list):
            # In a real ComfyUI node, printing to console is better than raising an exception
            # that might halt the whole queue. For now, adhering to plan.
            print(f"Error: Number of images ({len(image)}) does not match number of file paths ({len(paths_list)}).")
            # Consider how ComfyUI handles this. Returning () might be safest.
            # Or we can raise an error if that's preferred for user feedback.
            # Let's log and return empty, as ComfyUI might not show ValueError well.
            return ()


        os.makedirs(sfw_output_dir, exist_ok=True)
        os.makedirs(nsfw_output_dir, exist_ok=True)

        transform = T.ToPILImage()
        # It's generally better to initialize the pipeline once if possible, e.g., in __init__,
        # but for simplicity in this custom node structure, it's often here.
        classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection")

        for i in range(len(image)):
            img_tensor = image[i]
            original_path = paths_list[i]

            # The input 'image' tensor is for ComfyUI display/preview or further processing if needed.
            # For classification, we convert it to PIL.
            # For saving with metadata, we re-open the original file from disk.
            pil_img_for_classification = transform(img_tensor.permute(2, 0, 1))
            
            result = classifier(pil_img_for_classification)
            
            is_classified_nsfw = False
            if result: # Ensure result is not empty
                for r_item in result:
                    if r_item["label"] == "nsfw" and r_item["score"] >= score:
                        is_classified_nsfw = True
                        break
            
            target_dir = nsfw_output_dir if is_classified_nsfw else sfw_output_dir
            
            try:
                original_filename = os.path.basename(original_path)
                name, ext = os.path.splitext(original_filename)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S%f')
                output_filename = f"{timestamp}_{name}{ext}"
                output_filepath = os.path.join(target_dir, output_filename)

                # Load original image from disk to preserve metadata
                pil_image_to_save = Image.open(original_path)
                pil_image_to_save.load() # Ensure all image data and metadata are loaded

                metadata_params = {}
                if 'exif' in pil_image_to_save.info:
                    metadata_params['exif'] = pil_image_to_save.info['exif']
                if 'icc_profile' in pil_image_to_save.info:
                    metadata_params['icc_profile'] = pil_image_to_save.info['icc_profile']

                if pil_image_to_save.format == 'PNG':
                    pnginfo_data = PngImagePlugin.PngInfo()
                    # Try to copy text chunks for PNG workflow/metadata
                    if hasattr(pil_image_to_save, 'text') and pil_image_to_save.text:
                        for key, value in pil_image_to_save.text.items():
                            pnginfo_data.add_text(key, str(value)) # Ensure value is string
                    # Fallback for older Pillow versions or if .text is not populated but .info has items
                    elif pil_image_to_save.info:
                         for key, value in pil_image_to_save.info.items():
                            # Heuristic: many text chunks are strings. Avoid re-adding exif/icc.
                            if isinstance(key, str) and isinstance(value, str) and key not in ['exif', 'icc_profile', 'dpi']:
                                # Check if it's a known key that PIL might handle separately or if it's likely a custom tEXt chunk
                                if key.lower() not in ['gamma', 'srgb', 'chromaticity']: # Avoid some known non-textual metadata often in info
                                    try:
                                        pnginfo_data.add_text(key, value)
                                    except Exception as e_png_text:
                                        print(f"Warning: Could not add PngInfo text chunk for key '{key}': {e_png_text}")
                    if pnginfo_data.chunks: # Only add if we actually got some text data
                        metadata_params['pnginfo'] = pnginfo_data
                
                # Save the image (which is the original image re-opened from disk)
                pil_image_to_save.save(output_filepath, **metadata_params)
                # print(f"Saved: {output_filepath} (NSFW: {is_classified_nsfw})")

            except Exception as e:
                print(f"Error processing or saving image {original_path}: {e}")
        
        return ()


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "NSFWDetection": NSFWDetection
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "NSFWDetection": "NSFW Detection"
}
