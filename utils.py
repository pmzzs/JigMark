import torch
import numpy as np

import torch.nn as nn
import math
import random
import os
import numpy as np
import torch
import torchvision
from torchvision.transforms import v2
import PIL
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import json
import torch

def set_seeds(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

class ImageGenerator:
    def __init__(self, api_key) -> str:
        openai.api_key = api_key
        self.APIKey = openai.api_key
        
    def imageVariations(self, img_tensor, VariationCount = 1, ImageSize = "256x256"):
        def decode_base64_json_to_tensor(base64_json):
            totensor = transforms.ToTensor()
            base64_decoded = base64.b64decode(base64_json)
            image = Image.open(io.BytesIO(base64_decoded))
            return totensor(np.array(image))
        
        transform = transforms.ToPILImage()
        res_batch = torch.zeros((img_tensor.shape))
        for i in range(len(img_tensor)):
            pil_img = transform(img_tensor[i])
            byte_stream = BytesIO()
            pil_img.save(byte_stream, format='PNG')
            byte_array = byte_stream.getvalue()
            
            response = openai.Image.create_variation(
                image=byte_array,
                n=VariationCount,
                size=ImageSize,
                response_format = 'b64_json'
                )
            
            res_batch[i] = decode_base64_json_to_tensor(response['data'][0].b64_json)
            
        return res_batch

class ImageEditDataset(Dataset):
    def __init__(self, image_dataset, instruction_file, transform = None):
        self.image_dataset = image_dataset
        self.edit_instruction = self._load_json(instruction_file)
        self.transform = transform
        
    def _load_json(self, filename):
        with open(filename, 'r') as file:
            data = json.load(file)
        
        result = {}
        for entry in data:
            image_id = int(entry["Image ID"])
            # print(entry)
            edit_type = entry["Edit Type"]
            edit_instruction = entry["Edit Instruction"]
            edit_image_content = entry["Edit Image Content"]
            
            result[image_id] = (edit_type, edit_instruction, edit_image_content)
        
        return result

    def __getitem__(self, idx):
        image = self.image_dataset[idx][0]
        if self.transform is not None:
            image = self.transform(image)
        image_edit = self.edit_instruction[idx]
        return image, image_edit
    
    
    def __len__(self):
        return len(self.image_dataset)

def process_txt_file(filename):
    values = []
    
    # Read the file and parse the ID and content
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split(',', 1)
            value = parts[1][:75]  # truncate to 75 characters and add ellipsis
            values.append(value)
        
    return values

class ImageShuffler:
    def __init__(self, splits, shuffle_indices, transform_indices=None):
        self.splits = splits  # Number of splits per dimension
        self.shuffle_indices = shuffle_indices  # Fixed order for shuffling
        self.transform_indices = transform_indices  # Fixed order for transformations
        assert len(shuffle_indices) == splits * splits, "Invalid shuffle indices length"
        if transform_indices is not None:
            assert len(transform_indices) == splits * splits, "Invalid transform indices length"
    
        
    def transform_square(self, square, index, reverse=False):
        if index == 0:
            return torch.rot90(square, k=1 if not reverse else 3, dims=[1, 2])  # 90-degree rotation
        elif index == 1:
            return torch.rot90(square, k=2, dims=[1, 2])  # 180-degree rotation (same for reverse)
        elif index == 2:
            return torch.rot90(square, k=3 if not reverse else 1, dims=[1, 2])  # 270-degree rotation
        elif index == 3:
            return torch.flip(square, dims=(2,))  # horizontal flip (same for reverse)
        elif index == 4:
            return torch.flip(square, dims=(1,))  # vertical flip (same for reverse)
        else:
            raise ValueError(f'Invalid index: {index}')
    
    def random_exchange(self, arr, num_flips):
        # Check if the input for num_flips is valid
        if num_flips > len(arr) // 2:
            raise ValueError("num_flips cannot be greater than half the length of the array")

        # Create a set to keep track of indices that have been used
        used_indices = set()

        for _ in range(num_flips):
            # Randomly select two distinct indices
            while True:
                idx1, idx2 = random.sample(range(len(arr)), 2)
                # Ensure we haven't used these indices before
                if idx1 not in used_indices and idx2 not in used_indices:
                    break
            # Swap the elements at these indices
            arr[idx1], arr[idx2] = arr[idx2], arr[idx1]
            # Mark these indices as used
            used_indices.add(idx1)
            used_indices.add(idx2)

        return arr
    
    def shuffle(self, batch, update_shuffle_indices=False, mismatch_number = None):
        assert batch.size(2) == batch.size(3), "Images must be square"
        if update_shuffle_indices:
            if mismatch_number == None:
                self.shuffle_indices = [random.sample(range(self.splits**2), self.splits**2) for _ in range(batch.size(0))]
            else:
                self.shuffle_indices = [self.random_exchange(self.shuffle_indices[i], mismatch_number) for i in range(batch.size(0))]
            self.transform_indices = [[random.choice(range(5)) for _ in range(self.splits**2)] for _ in range(batch.size(0))]
        elif isinstance(self.shuffle_indices[0], int):
            self.shuffle_indices = [self.shuffle_indices for _ in range(batch.size(0))]
        
        batch_size = batch.size(0)
        side_length = batch.size(2) // self.splits
        multiplier = batch_size // len(self.shuffle_indices)
        
        recombined_images = []
        
        for k in range(multiplier):
            for b in range(len(self.shuffle_indices)):
                image = batch[b + k*len(self.shuffle_indices)]
                
                squares = [image[:, i*side_length:(i+1)*side_length, j*side_length:(j+1)*side_length]
                        for i in range(self.splits) for j in range(self.splits)]
                
                transformed_squares = [self.transform_square(sq, self.transform_indices[b][i]) for i, sq in enumerate(squares)]
                shuffled_squares = [transformed_squares[i] for i in self.shuffle_indices[b]]
                
                recombined_rows = [torch.cat(shuffled_squares[i:i+self.splits], dim=2) for i in range(0, self.splits * self.splits, self.splits)]
                recombined_image = torch.cat(recombined_rows, dim=1)
                
                recombined_images.append(recombined_image.unsqueeze(0))
        
        recombined_batch = torch.cat(recombined_images, dim=0)
        return recombined_batch
    
    def unshuffle(self, recombined_batch):
        assert recombined_batch.size(2) == recombined_batch.size(3), "Images must be square"
        
        side_length = recombined_batch.size(2) // self.splits
        
        original_images = []
        for b in range(len(self.shuffle_indices)):
            recombined_image = recombined_batch[b]
            indices = self.shuffle_indices[b]
            
            inverse_indices = torch.argsort(torch.tensor(indices))
            
            squares = [recombined_image[:, i*side_length:(i+1)*side_length, j*side_length:(j+1)*side_length]
                       for i in range(self.splits) for j in range(self.splits)]
            
            unshuffled_squares = [squares[i] for i in inverse_indices]
            inverse_transformed_squares = [self.transform_square(sq, self.transform_indices[b][i], reverse=True) for i, sq in enumerate(unshuffled_squares)]
            
            original_rows = [torch.cat(inverse_transformed_squares[i:i+self.splits], dim=2) for i in range(0, self.splits * self.splits, self.splits)]
            original_image = torch.cat(original_rows, dim=1)
            
            original_images.append(original_image.unsqueeze(0))
        
        original_batch = torch.cat(original_images, dim=0)
        return original_batch


class SiameseNetwork(nn.Module):
    def __init__(self, pretrained="ResNet50_Weights.IMAGENET1K_V2", transforms=False, freeze_backbone = True):
        super(SiameseNetwork, self).__init__()
        
        # Load the pre-trained weights of ResNet50
        self.resnet50 = torchvision.models.resnet50(weights=pretrained)
        
        if freeze_backbone:
            self.resnet50.eval()
            # Freeze the weights of the backbone
            for param in self.resnet50.parameters():
                param.requires_grad = False
        
        # Remove the last layer (fc layer) to get embeddings
        self.embedding_net = nn.Sequential(*list(self.resnet50.children())[:-1])
        
        # Flatten the output
        self.flatten = nn.Flatten()
        
        # Define the MLP
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),   # Assuming we're using ResNet50, embedding size is 2048
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid()           # Ensuring output is between 0 and 1
        )
        
        self.transforms = None
        if transforms:
            self.transforms = v2.Compose([
                v2.Resize((224, 224)),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def forward_one(self, x):
        x = self.embedding_net(x)
        x = self.flatten(x)
        return x

    def forward(self, input1, input2):
        if self.transforms is not None:
            input1 = self.transforms(input1)
            input2 = self.transforms(input2)
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        
        # Compute absolute difference between outputs
        combined = torch.abs(output1 - output2)
        similarity_score = self.fc(combined)
        
        return similarity_score

#Get results

def eval_res(ex_value, dex_value, dx_value, x_value):
    ex_values = torch.cat(ex_value).detach().cpu()
    dex_values = torch.cat(dex_value).detach().cpu()
    dx_values = torch.cat(dx_value).detach().cpu()
    x_values = torch.cat(x_value).detach().cpu()
    
    # Convert values to binary labels based on conditions
    ex_label = torch.ones_like(ex_values)
    dex_label = torch.ones_like(dex_values)
    dx_label = torch.zeros_like(dx_values)
    x_label = torch.zeros_like(x_values)
    # Combine the labels and scores
    all_labels = torch.cat([ex_label, dex_label, dx_label, x_label])
    all_scores = torch.cat([ex_values, dex_values, dx_values, x_values])  # Negate dx_value and x_value scores because you want them to be < 0

    # Compute TPR and FPR for various thresholds
    fpr, tpr, thresholds = roc_curve(all_labels.cpu().numpy(), all_scores.cpu().numpy())
    roc_auc = auc(fpr, tpr)

    # Find the highest TPR where FPR <= 1%
    filtered_indices = np.where(fpr <= 0.05)[0]
    max_tpr_index = filtered_indices[np.argmax(tpr[filtered_indices])]
    max_tpr_threshold = thresholds[max_tpr_index]

    print("TPR @ 1% FPR :", tpr[max_tpr_index])
    print("Corresponding Threshold:", max_tpr_threshold)
    print("AUC:", roc_auc)

    # Plot the ROC curve
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()

def adjust_learning_rate(optimizer, epoch, max_lr = 1e-3, min_lr = 1e-6, warmup_epochs = 3, total_epochs = 30):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < warmup_epochs:
        lr = max_lr * epoch / warmup_epochs 
    else:
        lr = min_lr + (max_lr - min_lr) * 0.5 * (1. + math.cos(math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

def contrastive_loss(ex, dex, dx, x, margin = 3, temperature = 0.1):
    pos = torch.cat((ex, dex),dim=0)
    neg = torch.cat((dx, x),dim=0)
    
    # If x extra low, make y high
    pos_loss = torch.mean(torch.log(1 + torch.exp((-pos + margin)/temperature)))
    
    # If x extra high, make y low
    neg_loss = torch.mean(torch.log(1 + torch.exp((neg + margin)/temperature)))

    return (pos_loss+neg_loss)

def max_gcd_less_than_32(num):
    max_gcd_val = 1
    for i in range(2, 33, 2):
        current_gcd = math.gcd(num, i)
        max_gcd_val = max(max_gcd_val, current_gcd)
    return max_gcd_val

def replace_batchnorm(module: torch.nn.Module):
    for name, child in module.named_children():
        if isinstance(child, torch.nn.BatchNorm2d):
            child: torch.nn.BatchNorm2d = child
            setattr(module, name, nn.GroupNorm(max_gcd_less_than_32(child.num_features), child.num_features))
        else:
            replace_batchnorm(child)

#Zero 123
def load_model_from_config(config, ckpt, device, verbose=False):
    config = OmegaConf.load(config)
    print(f'Loading model from {ckpt}')
    pl_sd = torch.load(ckpt, map_location='cpu')
    if 'global_step' in pl_sd:
        print(f'Global Step: {pl_sd["global_step"]}')
    sd = pl_sd['state_dict']
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print('missing keys:')
        print(m)
    if len(u) > 0 and verbose:
        print('unexpected keys:')
        print(u)

    model.to(device)
    model.eval()
    return model

@torch.no_grad()
def sample_model(input_im, model, sampler, precision, h, w, ddim_steps, n_samples, scale,
                 ddim_eta, x, y, z):
    precision_scope = autocast if precision == 'autocast' else nullcontext
    with precision_scope('cuda'):
        with model.ema_scope():
            c = model.get_learned_conditioning(input_im).tile(n_samples, 1, 1)
            T = torch.tensor([math.radians(x), math.sin(
                math.radians(y)), math.cos(math.radians(y)), z])
            T = T[None, None, :].repeat(n_samples, 1, 1).to(c.device)
            c = torch.cat([c, T], dim=-1)
            c = model.cc_projection(c)
            cond = {}
            cond['c_crossattn'] = [c]
            cond['c_concat'] = [model.encode_first_stage((input_im.to(c.device))).mode().detach()
                                .repeat(n_samples, 1, 1, 1)]
            if scale != 1.0:
                uc = {}
                uc['c_concat'] = [torch.zeros(n_samples, 4, h // 8, w // 8).to(c.device)]
                uc['c_crossattn'] = [torch.zeros_like(c).to(c.device)]
            else:
                uc = None

            shape = [4, h // 8, w // 8]
            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                             conditioning=cond,
                                             batch_size=n_samples,
                                             shape=shape,
                                             verbose=False,
                                             unconditional_guidance_scale=scale,
                                             unconditional_conditioning=uc,
                                             eta=ddim_eta,
                                             x_T=None)
            print(samples_ddim.shape)
            # samples_ddim = torch.nn.functional.interpolate(samples_ddim, 64, mode='nearest', antialias=False)
            x_samples_ddim = model.decode_first_stage(samples_ddim)
            return torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0).cpu()