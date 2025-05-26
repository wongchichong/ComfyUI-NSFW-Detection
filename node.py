# Use a pipeline as a high-level helper
from PIL import Image
from transformers import pipeline
import torchvision.transforms as T
import torch
import numpy # Kept for tensor2pil/pil2tensor


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
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("SFW_IMAGES", "NSFW_IMAGES")

    FUNCTION = "run"

    CATEGORY = "NSFWDetection"

    def run(self, image, score):
        transform = T.ToPILImage()
        classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection")

        sfw_image_list = []
        nsfw_image_list = []

        for i in range(len(image)):
            img_tensor = image[i] # This is a single image tensor from the batch
            pil_img = transform(img_tensor.permute(2, 0, 1))
            
            result = classifier(pil_img)
            
            is_classified_nsfw = False
            if result: # Ensure result is not empty
                for r_item in result: # Iterate over potentially multiple classifications
                    if r_item["label"] == "nsfw" and r_item["score"] >= score:
                        is_classified_nsfw = True
                        break # Found NSFW label that meets threshold
            
            if is_classified_nsfw:
                nsfw_image_list.append(img_tensor)
            else:
                sfw_image_list.append(img_tensor)

        # Prepare output tensors
        if sfw_image_list:
            sfw_images_tensor = torch.stack(sfw_image_list)
        else:
            # Return an empty tensor with shape (0, H, W, C) if dimensions of original batch are known
            # For now, using a simple empty tensor. This might need adjustment for ComfyUI.
            # Let's try to make it (0,H,W,C) if image is not empty, otherwise (0)
            if image.nelement() == 0 : # Check if input image batch itself is empty
                 sfw_images_tensor = torch.empty(0)
            else:
                 sfw_images_tensor = torch.empty((0, image.shape[1], image.shape[2], image.shape[3]), dtype=image.dtype, device=image.device)


        if nsfw_image_list:
            nsfw_images_tensor = torch.stack(nsfw_image_list)
        else:
            if image.nelement() == 0 :
                 nsfw_images_tensor = torch.empty(0)
            else:
                 nsfw_images_tensor = torch.empty((0, image.shape[1], image.shape[2], image.shape[3]), dtype=image.dtype, device=image.device)
                 
        return (sfw_images_tensor, nsfw_images_tensor)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "NSFWDetection": NSFWDetection
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "NSFWDetection": "NSFW Detection"
}
