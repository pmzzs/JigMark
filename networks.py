import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class Block(nn.Module):
    """ ConvNeXtV2 Block.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + x
        return x


class ConvNext(nn.Module):
	'''
	SENet, with BasicBlock and BottleneckBlock
	'''

	def __init__(self, in_channels, blocks, block_type="Block"):
		super(ConvNext, self).__init__()

		layers = [eval(block_type)(in_channels)] if blocks != 0 else []
		for _ in range(blocks - 1):
			layer = eval(block_type)(in_channels)
			layers.append(layer)

		self.layers = nn.Sequential(*layers)

	def forward(self, x):
		return self.layers(x)

class WMConvNext(nn.Module):
    '''
    Insert a watermark into an image
    '''

    def __init__(self, blocks=4, channels=128):
        super(WMConvNext, self).__init__()

        self.down_layer = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(channels),
        )
        self.mid_layer = ConvNext(channels, blocks=blocks)
        self.up_layer = nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2)
        self.mix_layer = nn.Conv2d(int(channels) + 3, 3, kernel_size=7, padding=3, stride=1)
        self.activation = nn.Hardsigmoid()
        
    def forward(self, image):
        x = self.down_layer(image)
        x = self.mid_layer(x)
        x = self.up_layer(x)
        concat = torch.cat([x, image], dim=1)
        output = self.mix_layer(concat)
        return self.activation(output)
    
import torch.nn as nn
import torch.nn.functional as F
import torch

class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class Block(nn.Module):
    """ ConvNeXtV2 Block.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + x
        return x

class LNConv(nn.Module):
	def __init__(self, channels_in, channels_out, kernel=3, stride=1, padding = 0):
		super(LNConv, self).__init__()

		self.layers = nn.Sequential(
			LayerNorm(channels_in, eps=1e-6, data_format="channels_first"),
			nn.Conv2d(channels_in, channels_out, kernel, stride, padding=padding),
		)

	def forward(self, x):
		return self.layers(x)

class ConvNext(nn.Module):
	'''
	SENet, with BasicBlock and BottleneckBlock
	'''

	def __init__(self, in_channels, blocks, block_type="Block"):
		super(ConvNext, self).__init__()

		layers = [eval(block_type)(in_channels)] if blocks != 0 else []
		for _ in range(blocks - 1):
			layer = eval(block_type)(in_channels)
			layers.append(layer)

		self.layers = nn.Sequential(*layers)

	def forward(self, x):
		return self.layers(x)
    
class ConvNextU(nn.Module):
    '''
    Insert a watermark into an image
    '''

    def __init__(self, blocks=2, channels=128, down_ratio = 2):
        super(ConvNextU, self).__init__()
        self.down1 = nn.Sequential(nn.Conv2d(3, channels, kernel_size=3, padding=1, stride=2),
                                   LayerNorm(128),
                                   ConvNext(channels, blocks=blocks))
        self.down2 = nn.Sequential(LNConv(channels, channels*2, kernel=4, stride=down_ratio*2),
                                   ConvNext(channels*2, blocks=blocks))
        self.down3 = nn.Sequential(LNConv(channels*2, channels*4, kernel=4, stride=down_ratio*2),
                                      ConvNext(channels*4, blocks=blocks))
        self.down4 = nn.Sequential(LNConv(channels*4, channels*8, kernel=4, stride=down_ratio*2),
                                      ConvNext(channels*8, blocks=blocks))
        
        self.up1 = nn.ConvTranspose2d(channels*2, channels, kernel_size=2, stride=down_ratio)
        self.up2 = nn.ConvTranspose2d(channels*4, channels, kernel_size=down_ratio*2, stride=down_ratio*2)
        self.up3 = nn.ConvTranspose2d(channels*8, channels*2, kernel_size=down_ratio*2, stride=down_ratio*2)
        self.up4 = nn.ConvTranspose2d(channels*8, channels*4, kernel_size=down_ratio*2, stride=down_ratio*2)
        
        self.mask = self.generate_mask(64)
        self.mix_layer = nn.Conv2d(int(channels) + 3, 3, kernel_size=7, padding=3, stride=1)
        
        self.activation = nn.Hardsigmoid()
        
    def generate_mask(self, block_size, tau=0.07):
        # Create coordinate grid
        x = torch.arange(0, block_size).float().unsqueeze(0).repeat(block_size, 1)
        y = torch.arange(0, block_size).float().unsqueeze(1).repeat(1, block_size)
        
        # Center coordinates
        center = block_size // 2
        
        # Compute distances to center
        dx = torch.abs(x - center)
        dy = torch.abs(y - center)
        
        # Get maximum of absolute distances
        max_dist = torch.max(dx, dy)
        
        # Normalize max_dist to [0, 1]``
        normalized_max_dist = max_dist / (block_size / 2)
        
        # Exponential mask
        mask = torch.exp(normalized_max_dist/tau)  # exp(0) = 1, so we subtract 1 to get 0 at the center
        mask = mask / mask.max()  # Normalize to [0, 1]
        mask = torch.where(mask < 0.01, torch.zeros_like(mask), mask)
        mask = 1 - mask.repeat(4, 4)
        
        return mask
        
    def forward(self, image):
        x1 = self.down1(image)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        
        x = self.up4(x4)
        x = self.up3(torch.cat([x, x3], dim=1))
        x = self.up2(torch.cat([x, x2], dim=1))
        x = self.up1(torch.cat([x, x1], dim=1))
        concat = torch.cat([x, image], dim=1)
        output = self.mix_layer(concat)
        
        if self.mask.get_device() == "cpu" or self.mask.get_device() < 0:
            self.mask = self.mask.to(x.get_device())
        
        return self.activation(output) * self.mask + image * (1 - self.mask)
    
def WMDecoder():
    decoder = torchvision.models.mobilenet_v3_large()
    proj_head = nn.Sequential(
            nn.Linear(960, 1280, bias=True),
            nn.Hardswish(),
            nn.Linear(1280, 1, bias=True)
        )
    decoder.classifier = proj_head
    return decoder