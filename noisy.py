import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import numpy as np
import math
import torchvision
import torch.nn.functional as F
import io
from torchvision.transforms import ToPILImage, ToTensor

import torch
import copy
from PIL import Image
import string
import random

class GaussianNoise:
    def __init__(self, mean, std_range=(0.05, 0.2)):
        """
        Initialize the GaussianNoise class.

        Args:
        - mean (float): Mean of the Gaussian noise.
        - std_range (tuple): A tuple containing the minimum and maximum 
                             values for the standard deviation (std) of the Gaussian noise.
        """
        self.mean = mean
        self.std_range = std_range

    def apply(self, images):
        """
        Add Gaussian noise with a randomly sampled std to the input batch of image tensors.

        Args:
        - images (torch.Tensor): Input batch of image tensors of shape 
                                 (batch_size, channels, height, width).

        Returns:
        - torch.Tensor: Batch of image tensors with added Gaussian noise.
        """
        std = random.uniform(*self.std_range)
        noise = torch.randn_like(images) * std + self.mean
        noisy_images = images + noise
        return noisy_images


class GaussianBlur:
    def __init__(self, kernel_size, sigma_range=(0.1, 2.0)):
        """
        Initialize the GaussianBlur class.

        Args:
        - kernel_size (int): Size of the Gaussian kernel. Should be odd.
        - sigma_range (tuple): A tuple containing the minimum and maximum 
                               values for the standard deviation (sigma) of the Gaussian kernel.
        """
        self.kernel_size = kernel_size
        self.sigma_range = sigma_range

    def _create_gaussian_kernel(self, kernel_size, sigma):
        """Create a Gaussian kernel."""
        x = np.linspace(-kernel_size // 2 + 1, kernel_size // 2, kernel_size)
        y = np.linspace(-kernel_size // 2 + 1, kernel_size // 2, kernel_size)
        x, y = np.meshgrid(x, y)
        kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        return torch.tensor(kernel, dtype=torch.float32)

    def apply(self, image):
        """
        Apply Gaussian blur to the input image tensor with a randomly sampled sigma.

        Args:
        - image (torch.Tensor): Input image tensor of shape 
                                (channels, height, width).

        Returns:
        - torch.Tensor: Blurred image tensor.
        """
        sigma = random.uniform(*self.sigma_range)
        kernel = self._create_gaussian_kernel(self.kernel_size, sigma)
        kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(image.shape[0], 1, 1, 1)  # Shape: [channels, 1, kernel_size, kernel_size]
        
        blurred_image = F.conv2d(image.unsqueeze(0), kernel, padding=self.kernel_size // 2, groups=image.shape[0])
        return blurred_image.squeeze(0)


class RandomHFlip:
    def __init__(self, p=0.5):
        """
        Initialize the RandomHFlip class.

        Args:
        - p (float): Probability of the image being horizontally flipped. Default is 0.5.
        """
        self.p = p

    def apply(self, image):
        """
        Randomly flip the image horizontally based on the probability.

        Args:
        - image (torch.Tensor): Input image tensor of shape 
                                (batch_size, channels, height, width).

        Returns:
        - torch.Tensor: Image tensor after potential horizontal flip.
        """
        if random.random() < self.p:
            return torch.flip(image, dims=[-1])  # Flip along the width dimension
        return image
    
class RandomVFlip:
    def __init__(self, p=0.5):
        """
        Initialize the RandomVFlip class.

        Args:
        - p (float): Probability of the image being vertically flipped. Default is 0.5.
        """
        self.p = p

    def apply(self, image):
        """
        Randomly flip the image vertically based on the probability.

        Args:
        - image (torch.Tensor): Input image tensor of shape 
                                (batch_size, channels, height, width).

        Returns:
        - torch.Tensor: Image tensor after potential vertical flip.
        """
        if random.random() < self.p:
            return torch.flip(image, dims=[-2])  # Flip along the height dimension
        return image
    
class ContrastAdjustment:
    def __init__(self, factor_range=(0.8, 1.2)):
        """
        Initialize the ContrastAdjustment class.

        Args:
        - factor_range (tuple): A tuple containing the minimum and maximum 
                                factors to adjust the contrast.
        """
        self.factor_range = factor_range

    def apply(self, image):
        """
        Randomly adjust the contrast of the image based on the factor range.

        Args:
        - image (torch.Tensor): Input image tensor of shape 
                                (batch_size, channels, height, width).

        Returns:
        - torch.Tensor: Image tensor with adjusted contrast.
        """
        factor = random.uniform(*self.factor_range)
        return image * factor
    
class RandomMask:
    def __init__(self, inpaint_dataset, mask_size_range=(80, 224), device = None):
        """
        Initialize the RandomMask class.

        Args:
        - mask_size_range (tuple): A tuple containing the minimum and maximum sizes for the random mask.
        """
        self.mask_size_range = mask_size_range
        self.inpaint_dataset = inpaint_dataset
        self.device = device

    def apply(self, image):
        """
        Randomly mask a part of the image and inpaint the masked area with the inpaint_image.

        Args:
        - image (torch.Tensor): Input image tensor of shape 
                                (batch_size, channels, height, width).
        - inpaint_image (torch.Tensor): Image tensor used for inpainting the masked area.

        Returns:
        - torch.Tensor: Image tensor with the masked area inpainted.
        """
        if self.device == "cpu" or image.get_device() < 0:
            return image
        
        inpaint_image = self.inpaint_dataset[random.randint(0, len(self.inpaint_dataset)-1)][0].to(self.device)
        
        # Ensure the inpaint_image is of the same size as the input image
        assert image.shape == inpaint_image.shape, "The inpaint_image must be of the same size as the input image."

        # Generate a random mask
        mask_size = random.randint(*self.mask_size_range)
        mask_x = random.randint(0, image.shape[1] - mask_size)
        mask_y = random.randint(0, image.shape[2] - mask_size)
        mask = torch.zeros_like(image)
        mask[:, mask_x:mask_x+mask_size, mask_y:mask_y+mask_size] = 1
        mask = mask.to(self.device)

        # Apply the mask and inpaint
        masked_image = image * (1 - mask) + inpaint_image * mask

        return masked_image
    
class BrightnessAdjustment:
    def __init__(self, value_range=(-0.2, 0.2)):
        """
        Initialize the BrightnessAdjustment class.

        Args:
        - value_range (tuple): A tuple containing the minimum and maximum 
                               values to adjust the brightness.
        """
        self.value_range = value_range

    def apply(self, image):
        """
        Randomly adjust the brightness of the image based on the value range.

        Args:
        - image (torch.Tensor): Input image tensor of shape 
                                (batch_size, channels, height, width).

        Returns:
        - torch.Tensor: Image tensor with adjusted brightness.
        """
        value = random.uniform(*self.value_range)
        return image + value
    


class RandomJPEGCompression:
    def __init__(self, quality_range=(10, 90)):
        """
        Initialize the RandomJPEGCompression class.

        Args:
        - quality_range (tuple): A tuple containing the minimum and maximum 
                                 quality factors for JPEG compression.
        """
        self.quality_range = quality_range
        self.to_pil = ToPILImage()
        self.to_tensor = ToTensor()

    def apply(self, image):
        """
        Randomly apply JPEG compression to the image based on the quality range.

        Args:
        - image (torch.Tensor): Input image tensor of shape 
                                (batch_size, channels, height, width).

        Returns:
        - torch.Tensor: Image tensor after JPEG compression.
        """
        quality = random.randint(*self.quality_range)
        
        if len(image.shape) == 4:
            image = image[0]
            
        pil_image = self.to_pil(image)
        with io.BytesIO() as buffer:
            pil_image.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            compressed_image = Image.open(buffer)
            return self.to_tensor(compressed_image)
        
class RandomCropAndResize:
    def __init__(self, output_size=256, crop_size=(128, 224)):
        """
        Initialize the RandomCropAndResize class.

        Args:
        - output_size (tuple or int, optional): Desired output size after cropping and resizing. 
                                               If int, a square output is assumed. Default: 256.
        - crop_size (tuple, optional): A tuple containing the minimum and maximum size 
                                       for the random crop before resizing. 
                                       Default: (128, 224).
        """
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size
        
        self.crop_size = crop_size

    def apply(self, image):
        """
        Randomly crop the image and resize it back to the original size.

        Args:
        - image (torch.Tensor): Input image tensor of shape 
                                (channels, height, width).

        Returns:
        - torch.Tensor: Cropped and resized image tensor.
        """
        c, h, w = image.shape

        # Randomly choose a crop size within the specified range
        crop_h = crop_w = random.randint(*self.crop_size)

        if h < crop_h or w < crop_w:
            raise ValueError("Crop size should be smaller or equal to the original image size")

        top = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)

        cropped_image = image[:, top:top + crop_h, left:left + crop_w]

        # Resize the cropped image back to the original size
        return torchvision.transforms.functional.resize(cropped_image, size=self.output_size, antialias=True)
        
class Img2imgTransforms:
    def __init__(self, pipe, strength_range=(0.0, 1.0), guidance_scale_range=(1.0, 2.0), inference_steps_range=(20,50), gpu_id=7, num_images_per_prompt=1):
        """
        Initialize the Img2imgTransforms class.

        Args:
        - strength_range (tuple): A tuple containing the minimum and maximum 
                                  values for the strength.
        - guidance_scale_range (tuple): A tuple containing the minimum and maximum 
                                       values for the guidance_scale.
        - num_inference_steps (int): Number of inference steps.
        """
        self.strength_range = strength_range
        self.guidance_scale_range = guidance_scale_range
        self.inference_steps_range = inference_steps_range
        self.pipe = pipe
        self.to_tensor = ToTensor()
        self.num_images_per_prompt = num_images_per_prompt

    def apply(self, image_batch, prompts):
        """
        Apply the image transformation based on the given parameters.

        Args:
        - image_batch (torch.Tensor): Input image tensor of shape 
                                      (batch_size, channels, height, width).
        - prompts (list): List of prompts with the same length as image_batch.

        Returns:
        - torch.Tensor: Transformed image tensor.
        """
        strength = random.uniform(*self.strength_range)
        guidance_scale = random.uniform(*self.guidance_scale_range)
        num_inference_steps = random.randint(self.inference_steps_range[0], self.inference_steps_range[1])
        
        # Assuming the 'pipe' function is available in the current scope
        transformed_images = self.pipe(prompt=prompts, 
                                        image=image_batch.cpu(), 
                                        strength=strength, 
                                        guidance_scale=guidance_scale, 
                                        num_inference_steps=num_inference_steps,
                                        num_images_per_prompt=self.num_images_per_prompt).images
        
        return torch.stack([self.to_tensor(image) for image in transformed_images])
    
class RandomImageTransforms:
    def __init__(self, transforms, img2img_transforms, range=[0,1]):
        """
        Initialize the RandomImageTransforms class.

        Args:
        - transforms (list): List of image transformation classes.
        - max_transforms (int): Maximum number of transformations to apply on each call.
        """
        self.transforms = transforms
        self.range = range
        self.img2img_transforms = img2img_transforms
        
        
    def transform(self, images, prompts):
        """
        Randomly apply a subset of transformations to each image in the batch.

        Args:
        - images (torch.Tensor): Input batch of image tensors of shape 
                                 (batch_size, channels, height, width).     
        - prompts (list): List of prompts with the same length as image_batch.

        Returns:
        - torch.Tensor: Batch of transformed image tensors.
        """
        new_images = images
        current_device = images.get_device()
        transformed_images = self.img2img_transforms.apply(images, prompts)
        if current_device != -1:
            transformed_images = transformed_images.to(current_device)
        combined_images = torch.cat((new_images,transformed_images), dim=0)

        transformed_images_list = []
        for image in combined_images:
            if random.choice([0, 1]) > 0:
                selected_transforms = random.sample(self.transforms, 1)
                for transform in selected_transforms:
                    image = transform.apply(image)
                    image = torch.clamp(image, self.range[0], self.range[1])
                transformed_images_list.append(image.to(current_device))
            else:
                transformed_images_list.append(image.to(current_device))

        return torch.stack(transformed_images_list, dim=0)

    # def transform(self, images, prompts, selected_indices):
    #     """
    #     Randomly apply a subset of transformations to each image in the batch.

    #     Args:
    #     - images (torch.Tensor): Input batch of image tensors of shape 
    #                              (batch_size, channels, height, width).     
    #     - prompts (list): List of prompts with the same length as image_batch.

    #     Returns:
    #     - torch.Tensor: Batch of transformed image tensors.
    #     """
    #     new_images = images
    #     current_device = images.get_device()
    #     # Extract the selected images and prompts
    #     selected_images = new_images[selected_indices]
    #     selected_prompts = [prompts[i] for i in selected_indices]
    #     transformed_images = self.img2img_transforms.apply(selected_images, selected_prompts)
    #     if current_device != -1:
    #         transformed_images = transformed_images.to(current_device)
    #     new_images[selected_indices] = transformed_images

    #     transformed_images_list = []
    #     for image in new_images:
    #         num_transforms = random.randint(1, self.max_transforms)
    #         selected_transforms = random.sample(self.transforms, num_transforms)
    #         for transform in selected_transforms:
    #             image = transform.apply(image)
    #             image = torch.clamp(image, self.range[0], self.range[1])
    #         transformed_images_list.append(image.to(current_device))

    #     return torch.stack(transformed_images_list, dim=0)