import sys
import os
import torch
import torch.nn as nn



class SingleDeconv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.block = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        return self.block(x)
     
class SingleConv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super().__init__()
        self.block = nn.Conv2d(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2
        )

    def forward(self, x):
        return self.block(x)

class Conv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleConv2DBlock(in_planes, out_planes, kernel_size),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)

class Deconv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleDeconv2DBlock(in_planes, out_planes),
            SingleConv2DBlock(out_planes, out_planes, kernel_size),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)
    
def reshape_transformer_output(z, embed_dim, h, w, patch_size):
    # print(f"H, W : {h, w}")
    # print(f"patch_size : {patch_size}")
    B, N, C = z.shape  # B: 배치, N: 토큰 수, C: 채널
    num_patches = (h // patch_size) * (w // patch_size)
    # print(f"B, N, C : {B, N, C}")
    # print(f"num_patches : {num_patches}")
    if N == num_patches + 1: #CLS 토큰이 있는 경우
         z = z[:, 1:, :]  # CLS 토큰 제거
    h_patches = h // patch_size
    w_patches = w // patch_size
    
    return z.transpose(-1, -2).reshape(B, embed_dim, h_patches, w_patches)




# === 2D Decoder === #
class UNETRDecoder1(nn.Module):
    def __init__(self, embed_dim=768, patch_size = 14, input_dim=3, output_dim=3):
        super().__init__()

        self.input_dim = input_dim #4 
        self.output_dim = output_dim #3 
        self.embed_dim = embed_dim #768
        self.patch_size = patch_size #16
        self.num_layers = 12
        self.ext_layers = [3, 6, 9, 12]
        
       # Decoder layers
        self.decoder0 = nn.Sequential(
            Conv2DBlock(input_dim, 32, 3),
            Conv2DBlock(32, 64, 3)
        )

        self.decoder3 = nn.Sequential(
            Deconv2DBlock(embed_dim, 512),
            Deconv2DBlock(512, 256),
            Deconv2DBlock(256, 128)
        )

        self.decoder6 = nn.Sequential(
            Deconv2DBlock(embed_dim, 512),
            Deconv2DBlock(512, 256)
        )

        self.decoder9 = Deconv2DBlock(embed_dim, 512)

        self.decoder12_upsampler = SingleDeconv2DBlock(embed_dim, 512)

        self.decoder9_upsampler = nn.Sequential(
            Conv2DBlock(1024, 512),
            Conv2DBlock(512, 512),
            SingleDeconv2DBlock(512, 256)
        )

        self.decoder6_upsampler = nn.Sequential(
            Conv2DBlock(512, 256),
            Conv2DBlock(256, 256),
            SingleDeconv2DBlock(256, 128)
        )

        self.decoder3_upsampler = nn.Sequential(
            Conv2DBlock(256, 128),
            Conv2DBlock(128, 128),
            SingleDeconv2DBlock(128, 64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1) 
            
        )

        self.decoder0_header = nn.Sequential(
            Conv2DBlock(128, 64),
            Conv2DBlock(64, 64),
            SingleConv2DBlock(64, output_dim, 1)
        )

    def forward(self, x, features):
        z0, z3, z6, z9, z12 = x, *features
        B, C, H, W = x.shape

        z3 = reshape_transformer_output(z3, self.embed_dim, H, W, self.patch_size)
        z6 = reshape_transformer_output(z6, self.embed_dim, H, W, self.patch_size)
        z9 = reshape_transformer_output(z9, self.embed_dim, H, W, self.patch_size)
        z12 = reshape_transformer_output(z12, self.embed_dim, H, W, self.patch_size)

        # Decoder operations
        z12 = self.decoder12_upsampler(z12)  # 512
        z9 = self.decoder9(z9)  # 512
        z9 = self.decoder9_upsampler(torch.cat([z9, z12], dim=1))  # 1024 -> 512 -> 256
        z6 = self.decoder6(z6)  # 256
        z6 = self.decoder6_upsampler(torch.cat([z6, z9], dim=1))  # 512 -> 256 -> 128
        z3 = self.decoder3(z3)  # 128
        z3 = self.decoder3_upsampler(torch.cat([z3, z6], dim=1))  # 256 -> 128 -> 64
        z0 = self.decoder0(z0)  # 64
        output = self.decoder0_header(torch.cat([z0, z3], dim=1))  # 128 -> 64 -> 3

        return output
    
    
class UNETRDecoder2(nn.Module): 
    def __init__(self, embed_dim=768, patch_size = 16, input_dim=3, output_dim=3):
        super().__init__()

        self.input_dim = input_dim #4 
        self.output_dim = output_dim #3 
        self.embed_dim = embed_dim #768
        self.patch_size = patch_size #16
        self.num_layers = 12
        self.ext_layers = [3, 6, 9, 12]
        
       # Decoder layers
        self.decoder0 = nn.Sequential(
            Conv2DBlock(input_dim, 32, 3),
            Conv2DBlock(32, 64, 3)
        )

        self.decoder3 = nn.Sequential(
            Deconv2DBlock(embed_dim, 512),
            Deconv2DBlock(512, 256),
            Deconv2DBlock(256, 128)
        )

        self.decoder6 = nn.Sequential(
            Deconv2DBlock(embed_dim, 512),
            Deconv2DBlock(512, 256)
        )

        self.decoder9 = Deconv2DBlock(embed_dim, 512)

        self.decoder12_upsampler = SingleDeconv2DBlock(embed_dim, 512)

        self.decoder9_upsampler = nn.Sequential(
            Conv2DBlock(1024, 512),
            Conv2DBlock(512, 512),
            SingleDeconv2DBlock(512, 256)
        )

        self.decoder6_upsampler = nn.Sequential(
            Conv2DBlock(512, 256),
            Conv2DBlock(256, 256),
            SingleDeconv2DBlock(256, 128)
        )

        self.decoder3_upsampler = nn.Sequential(
            Conv2DBlock(256, 128),
            Conv2DBlock(128, 128),
            SingleDeconv2DBlock(128, 64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1) 
            
        )

        self.decoder0_header = nn.Sequential(
            Conv2DBlock(128, 64),
            Conv2DBlock(64, 64),
            SingleConv2DBlock(64, output_dim, 1)
        )

    def forward(self, x, features):
        z0, z3, z6, z9, z12 = x, *features
        B, C, H, W = x.shape

        z3 = reshape_transformer_output(z3, self.embed_dim, H, W, self.patch_size)
        z6 = reshape_transformer_output(z6, self.embed_dim, H, W, self.patch_size)
        z9 = reshape_transformer_output(z9, self.embed_dim, H, W, self.patch_size)
        z12 = reshape_transformer_output(z12, self.embed_dim, H, W, self.patch_size)

        # Decoder operations
        z12 = self.decoder12_upsampler(z12)  # 512
        z9 = self.decoder9(z9)  # 512
        z9 = self.decoder9_upsampler(torch.cat([z9, z12], dim=1))  # 1024 -> 512 -> 256
        z6 = self.decoder6(z6)  # 256
        z6 = self.decoder6_upsampler(torch.cat([z6, z9], dim=1))  # 512 -> 256 -> 128
        z3 = self.decoder3(z3)  # 128
        z3 = self.decoder3_upsampler(torch.cat([z3, z6], dim=1))  # 256 -> 128 -> 64
        z0 = self.decoder0(z0)  # 64
        output = self.decoder0_header(torch.cat([z0, z3], dim=1))  # 128 -> 64 -> 3

        return output   
    
class SAMDecoder(nn.Module):
    def __init__(self, embed_dim=768, patch_size = 16, input_dim=3, output_dim=3):
        super(SAMDecoder,self).__init__()
        self.input_dim = input_dim #4 
        self.output_dim = output_dim #3 
        self.embed_dim = embed_dim #768
        self.patch_size = patch_size #16
        self.num_layers = 12
        self.ext_layers = [3, 6, 9, 12]
        
        
        # Decoder layers
        self.decoder0 = nn.Sequential(
            Conv2DBlock(input_dim, 32, 3),
            Conv2DBlock(32, 64, 3)
        )

        self.decoder3 = nn.Sequential(
            Deconv2DBlock(embed_dim, 512),
            Deconv2DBlock(512, 256),
            Deconv2DBlock(256, 128)
        )

        self.decoder6 = nn.Sequential(
            Deconv2DBlock(embed_dim, 512),
            Deconv2DBlock(512, 256)
        )

        self.decoder9 = Deconv2DBlock(embed_dim, 512)

        self.decoder12_upsampler = SingleDeconv2DBlock(embed_dim, 512)

        self.decoder9_upsampler = nn.Sequential(
            Conv2DBlock(1024, 512),
            Conv2DBlock(512, 512),
            SingleDeconv2DBlock(512, 256)
        )

        self.decoder6_upsampler = nn.Sequential(
            Conv2DBlock(512, 256),
            Conv2DBlock(256, 256),
            SingleDeconv2DBlock(256, 128)
        )

        self.decoder3_upsampler = nn.Sequential(
            Conv2DBlock(256, 128),
            Conv2DBlock(128, 128),
            SingleDeconv2DBlock(128, 64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1) 
            
        )

        self.decoder0_header = nn.Sequential(
            Conv2DBlock(128, 64),
            Conv2DBlock(64, 64),
            SingleConv2DBlock(64, output_dim, 1)
        )

    def forward(self, x, features):
        z0, z3, z6, z9, z12 = x, *features
        B, C, H, W = x.shape

        z3 = reshape_transformer_output(z3, self.embed_dim, H, W, self.patch_size)
        z6 = reshape_transformer_output(z6, self.embed_dim, H, W, self.patch_size)
        z9 = reshape_transformer_output(z9, self.embed_dim, H, W, self.patch_size)
        z12 = reshape_transformer_output(z12, self.embed_dim, H, W, self.patch_size)

        # Decoder operations
        z12 = self.decoder12_upsampler(z12)  # 512
        z9 = self.decoder9(z9)  # 512
        z9 = self.decoder9_upsampler(torch.cat([z9, z12], dim=1))  # 1024 -> 512 -> 256
        z6 = self.decoder6(z6)  # 256
        z6 = self.decoder6_upsampler(torch.cat([z6, z9], dim=1))  # 512 -> 256 -> 128
        z3 = self.decoder3(z3)  # 128
        z3 = self.decoder3_upsampler(torch.cat([z3, z6], dim=1))  # 256 -> 128 -> 64
        z0 = self.decoder0(z0)  # 64
        output = self.decoder0_header(torch.cat([z0, z3], dim=1))  # 128 -> 64 -> 3

        return output   
        
        
        
# === 2D 통합 모델 === #
class DummyEncoder(nn.Module):
    def __init__(self, batch_size=1, num_tokens=197, embed_dim=768):
        super(DummyEncoder, self).__init__()
        self.batch_size = batch_size
        self.num_tokens = num_tokens  # CLS 토큰 + 패치 토큰 (256 + 1)
        self.embed_dim = embed_dim    # 임베딩 차원 (768)
        
    def forward(self, x):
        """
        입력 x는 Decoder에서 첫 번째 입력(z0)으로 사용됩니다.
        """
        # 더미 feature map 생성
        z3 = torch.randn(self.batch_size, self.num_tokens, self.embed_dim)
        z6 = torch.randn(self.batch_size, self.num_tokens, self.embed_dim)
        z9 = torch.randn(self.batch_size, self.num_tokens, self.embed_dim)
        z12 = torch.randn(self.batch_size, self.num_tokens, self.embed_dim)
        
        return [z3, z6, z9, z12]
class UNETR(nn.Module):
    def __init__(self, encoder, embed_dim=768, patch_size = 16, input_dim=3, output_dim=3):
        super().__init__()
        self.encoder = encoder
        self.decoder = SAMDecoder(embed_dim,patch_size, input_dim, output_dim)

    def forward(self, x):
        features = self.encoder(x)
        print(f'encoder_features : {len(features)}, {features[0].shape}')
        output = self.decoder(x, features)
        return output
   
#Text code
if __name__ == '__main__':
    model = UNETR(encoder=DummyEncoder( num_tokens = 197),patch_size = 16)
    x = torch.randn(1, 3, 224, 224)
    print(f"inpt shape : {x.shape}")
    features = model(x)
    print(f"output shape : {features.shape}")
