import argparse
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.nn as nn
import numpy as np
import torchvision
from tqdm import tqdm
import torch
from utils import *
from noisy import *
from frn import *
from accelerate import Accelerator
from lpips_pytorch import LPIPS
import os
import wandb
from unet import ConvUNext3, ConvUNext4

from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler

save_path = "./checkpoints"

def main(encoder, decoder, dataloader, edit_captions, args): 
    inference_steps = (35,50)
    accelerator = Accelerator(log_with="wandb")
    current_name = ""
    
    accelerator.init_trackers(
        project_name=proj_name, 
        config={"learning_rate": learning_rate, 
                "inference_steps": inference_steps,
                "other_params":other_params}
    )        
    encoder_opt = torch.optim.AdamW(encoder.parameters(), lr=0.1, betas=(0.9, 0.999), weight_decay=0.05)
    decoder_opt = torch.optim.AdamW(decoder.parameters(), lr=0.1, betas=(0.9, 0.999), weight_decay=0.05)
    
    encoder, decoder, encoder_opt, decoder_opt, dataloader = accelerator.prepare(encoder, decoder, encoder_opt, decoder_opt, dataloader)
    encoder_clipper = AdaptiveGradientClipper()
    decoder_clipper = AdaptiveGradientClipper()
    
    num_of_splits = args.jigsaw_width
    shuffle_indices = random.sample(range(num_of_splits**2), num_of_splits**2)
    shuffler = ImageShuffler(splits=num_of_splits, shuffle_indices=shuffle_indices)
    
    loss_fn_vgg = LPIPS(
        net_type='vgg',  # choose a network type from ['alex', 'squeeze', 'vgg']
        version='0.1',  # Currently, v0.1 is supported,
        device = accelerator.device
    )
    smoothL1 = nn.SmoothL1Loss(beta=0.1)
    
    model_id_or_path = "lambdalabs/miniSD-diffusers"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16, requires_safety_checker = False)
    pipe.safety_checker = None
    pipe.feature_extractor = None
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    pipe = pipe.to(accelerator.device)
    

    encoder.train()
    decoder.train()
    for epoch in tqdm(range(args.total_epochs)):
        factor = min(1, 0.2+epoch/30)
        
        h_flipper = RandomHFlip()
        v_flipper = RandomVFlip()
        contrast_adjuster = ContrastAdjustment(factor_range=(0.8*factor, 1.5*factor))
        brightness_adjuster = BrightnessAdjustment(value_range=(-0.25*factor, 0.25*factor))
        jpeg_compressor = RandomJPEGCompression(quality_range=(int(10*factor), int(90*factor)))
        noiser = GaussianNoise(mean=0)
        blurer = GaussianBlur(kernel_size=7)
        Cropper = RandomCropAndResize(crop_size=((256-int(128*factor)),(256-int(32*factor))))
        img2img_editer = Img2imgTransforms(pipe = pipe, strength_range=(0.1+0.2*factor, 0.2+0.4*factor), guidance_scale_range=(5, 20), inference_steps_range = inference_steps,num_images_per_prompt=3)

        # Initialize the RandomImageTransforms class
        random_transformer = RandomImageTransforms(
            transforms=[h_flipper, v_flipper, noiser, blurer, contrast_adjuster, brightness_adjuster, jpeg_compressor, Cropper],
            img2img_transforms = img2img_editer
        )
        for iters, (img, _) in enumerate(dataloader):
            len_loader = len(dataloader)
            adjust_learning_rate(encoder_opt, epoch+iters/len_loader-70, max_lr=1e-4)
            adjust_learning_rate(decoder_opt, epoch+iters/len_loader-70, max_lr=1e-4)
            # Shuffle the image and watermark the shuffule version, then unshuffle it
            w_img = shuffler.unshuffle(encoder(shuffler.shuffle(img, update_shuffle_indices=True)))
            
            '''
			train encoder and decoder
			'''
            batch_size = img.shape[0]
                        
            #Diffusion model
            prompts = random.sample(edit_captions, img.shape[0])

            # Image recover loss
            lpips_loss = loss_fn_vgg(w_img,img)/batch_size
            smoothl1_loss = smoothL1(w_img, img)
            image_loss = 1.5 * lpips_loss + 500 * smoothl1_loss
                        
            with torch.no_grad():
                diff_img = random_transformer.transform(w_img.detach(), prompts)
                d_diff_img = random_transformer.transform(img.detach(), prompts)
            
            ori_decode = decoder(shuffler.shuffle(img)) # change to img
            ori_diff_decode = decoder(shuffler.shuffle(d_diff_img).to(accelerator.device))
            encode_decode = decoder(shuffler.shuffle(w_img)) # change to wimg
            diff_decode = decoder(shuffler.shuffle(diff_img).to(accelerator.device))
            
            # Negative mismatch sample
            mismatch_num = random.randint(1, num_of_splits*num_of_splits/2)
            # Return to w_img
            ex_parts = decoder(shuffler.shuffle(w_img, update_shuffle_indices=True, mismatch_number=mismatch_num)) # change to wimg
            dex_parts = decoder(shuffler.shuffle(diff_img.to(accelerator.device)))
            ori_diff_decode = torch.cat((ori_diff_decode, ex_parts, dex_parts), dim=0)
            
            # Get loss
            nce_loss = contrastive_loss(encode_decode, diff_decode, ori_diff_decode, ori_decode)
            new_loss = nce_loss + image_loss            
            
            #Backward
            accelerator.wait_for_everyone()
            accelerator.backward(new_loss)
            if accelerator.sync_gradients:
                encoder_clipper.clip_gradients(encoder, accelerator, accele=False)
                decoder_clipper.clip_gradients(decoder, accelerator, accele=False)
            
            #Step
            encoder_opt.step()
            decoder_opt.step()
            
            encoder.zero_grad()
            decoder.zero_grad()
            
            if accelerator.sync_gradients and accelerator.is_main_process:
                accelerator.print(lpips_loss.item(), mse_loss.item())
                accelerator.print(mismatch_num, torch.mean(ex_parts).item(), torch.mean(dex_parts).item())
                current_name = wandb.run.name
                
                ex_value, dex_value, dx_value, x_value = torch.mean(encode_decode), torch.mean(diff_decode), torch.mean(ori_diff_decode), torch.mean(ori_decode)
                log_x, log_ex, log_dex, log_dx = wandb.Image(img[0]), wandb.Image(w_img[0]), wandb.Image(diff_img[batch_size]), wandb.Image(d_diff_img[batch_size])
                accelerator.print(f"Iters: {iters}, nce_loss: {nce_loss.item():.5f}, image_loss: {image_loss.item():.5f}, w_diff_cr: {w_diff_cr:.5f}, o_diff_cr: {o_diff_cr:.5f}, total_cr: {total_cr:.5f}")
                accelerator.log({"nce_loss": nce_loss.item(), "image_loss": image_loss.item(), "ex_value": ex_value, "dex_value": dex_value, "dx_value": dx_value, "x_value": x_value, "x_img": log_x,\
                            "ex_img": log_ex, "dex_img": log_dex, "dx_img": log_dx, "lr": encoder_opt.param_groups[0]['lr'], "ACC": total_cr})
            if iters % 100 == 0 and iters > 10 and accelerator.is_main_process:
                accelerator.save({
                    "epoch": epoch,
                    "step": iters,
                    "encoder_opt": encoder_opt.state_dict(),
                    "decoder_opt": decoder_opt.state_dict(),    
                    "encoder": accelerator.unwrap_model(encoder).state_dict(),
                    "decoder": accelerator.unwrap_model(decoder).state_dict(),
                }, save_path + "/" + current_name + "_models.pth")
                print("Finish save on path:" + save_path)
                
        # End of epoch            
        accelerator.print("epoch:" + str(epoch))
        accelerator.save_state(save_path + "/" + save_path)
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training script for a model.')
    parser.add_argument('--train_path', type=str, default='/home/data/imagenet/test', help='Path to the training data.')
    parser.add_argument('--instr_path', type=str, default='./dataset/edit_caption.txt', help='Path to the instruction data.')
    parser.add_argument('--img_size', type=int, default=256, help='The size of the images.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--total_epochs', type=int, default=10, help='Total number of training epochs.')
    parser.add_argument('--jigsaw_width', type=int, default=4, help='Number of jigsaw pieces in the width.')

    args = parser.parse_args()
    
    train_transform = transforms.Compose([
                transforms.Resize((args.img_size,args.img_size), antialias=True),
                transforms.ToTensor(),
            ])
    
    trainset = torchvision.datasets.ImageFolder(root=args.train_path, transform=train_transform)
    edit_captions = process_txt_file(args.instr_path)
    trainset_dataloader = torch.utils.data.DataLoader(trainset, batch_size=48, shuffle=True)
    
    encoder = ConvUNext3()
    decoder = torchvision.models.mobilenet_v3_large()
    proj_head = nn.Sequential(
            nn.Linear(960, 1280, bias=True),
            nn.Hardswish(),
            nn.Linear(1280, 1, bias=True)
        )
    decoder.classifier = proj_head
    replace_batchnorm(decoder)
    
    main(encoder, decoder, trainset_dataloader, edit_captions, args)