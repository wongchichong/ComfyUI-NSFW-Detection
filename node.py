# Use a pipeline as a high-level helper
from PIL import Image
from transformers import pipeline
import torchvision.transforms as T
import torch
import numpy
import os
from datetime import datetime


def tensor2pil(image):
    return Image.fromarray(numpy.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(numpy.uint8))


def pil2tensor(image):
    return torch.from_numpy(numpy.array(image).astype(numpy.float32) / 255.0).unsqueeze(0)


class NSFWDetection:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "score": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "round": 0.001,
                    # The value represeting the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
                    "display": "nsfw_threshold"}),
                "alternative_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "run"

    CATEGORY = "NSFWDetection"

    def run(self, image, score, alternative_image):
        transform = T.ToPILImage()
        classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection")

        sfw_dir = "output/sfw"
        nsfw_dir = "output/nsfw"
        os.makedirs(sfw_dir, exist_ok=True)
        os.makedirs(nsfw_dir, exist_ok=True)

        for i in range(len(image)):
            original_image_pil = transform(image[i].permute(2, 0, 1))
            result = classifier(original_image_pil)
            image_size = image[i].size()
            width, height = image_size[1], image_size[0]

            unique_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S%f')}.png"

            is_nsfw = False
            if result: # Ensure result is not empty
                for r in result:
                    if r["label"] == "nsfw" and r["score"] > score:
                        is_nsfw = True
                        # Save original NSFW image
                        nsfw_image_path = os.path.join(nsfw_dir, unique_filename)
                        original_image_pil.save(nsfw_image_path)

                        # Save alternative SFW image
                        alternative_image_pil = transform(alternative_image[0].permute(2, 0, 1))
                        sfw_alternative_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S%f')}_alt.png"
                        sfw_alternative_image_path = os.path.join(sfw_dir, sfw_alternative_filename)
                        alternative_image_pil.resize((width, height), resample=Image.Resampling(2)).save(sfw_alternative_image_path)
                        
                        # Replace original image with alternative
                        image[i] = pil2tensor(alternative_image_pil.resize((width, height), resample=Image.Resampling(2)))
                        break # Found NSFW label, no need to check other results for this image
            
            if not is_nsfw:
                # Save original SFW image
                sfw_image_path = os.path.join(sfw_dir, unique_filename)
                original_image_pil.save(sfw_image_path)

        return (image,)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "NSFWDetection": NSFWDetection
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "NSFWDetection": "NSFW Detection"
}
