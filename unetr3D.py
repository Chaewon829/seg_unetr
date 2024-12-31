import decoder as decoder
from decoder_3D import UNETRDecoder3D,UNETRDecoder3D_14, SAMDecoder3D
from encoder import Dinov2Encoder, BLIPEncoder, OpenClipEncoder ,BEiTv2Encoder, SamEncoder
import sys
import os
import torch
import torch.nn as nn

class DINOUNETR(nn.Module):
    def __init__(self, embed_dim = 768, patch_size = 14, input_dim = 3, output_dim=3):
        super(DINOUNETR, self).__init__()
        
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.encoder = Dinov2Encoder()
        self.decoder = UNETRDecoder3D_14(embed_dim=self.embed_dim, patch_size=self.patch_size, input_dim=self.input_dim, output_dim=self.output_dim)
        
        # Freeze encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
            
    def forward(self, x):

        B, D, C, H, W = x.shape
        x = x.view(B * D, C, H, W).contiguous()
        features = self.encoder(x)        
        # print(f'encoder_features : {len(features)}, {features[0].shape}')       
        reshaped_features = [
            f.view(B, D, f.shape[1], f.shape[2]).contiguous() for f in features
        ]
        output = self.decoder(x.view(B, D, C, H, W).contiguous(), reshaped_features)
        return output
    
    
class BeiTUNETR(nn.Module):
    def __init__(self, embed_dim = 768, patch_size = 16, input_dim = 3, output_dim=3):
        super(BeiTUNETR, self).__init__()
        
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.encoder = BEiTv2Encoder()
        self.decoder = UNETRDecoder3D(embed_dim=self.embed_dim, patch_size=self.patch_size, input_dim=self.input_dim, output_dim=self.output_dim) 

        # Freeze encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
            
    def forward(self, x):

        B, D, C, H, W = x.shape
        x = x.view(B * D, C, H, W).contiguous()
        features = self.encoder(x)        
        # print(f'encoder_features : {len(features)}, {features[0].shape}')       
        reshaped_features = [
            f.view(B, D, f.shape[1], f.shape[2]).contiguous() for f in features
        ]
        output = self.decoder(x.view(B, D, C, H, W).contiguous(), reshaped_features)
        return output
    
class BLIPUNETR(nn.Module):
    def __init__(self, embed_dim = 768, patch_size = 16, input_dim = 3, output_dim=3):
        super(BLIPUNETR, self).__init__()
        
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.encoder = BLIPEncoder()
        self.decoder = UNETRDecoder3D(embed_dim=self.embed_dim, patch_size=self.patch_size, input_dim=self.input_dim, output_dim=self.output_dim)

        # Freeze encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
            
    def forward(self, x):

        B, D, C, H, W = x.shape
        x = x.view(B * D, C, H, W).contiguous()
        features = self.encoder(x)        
        # print(f'encoder_features : {len(features)}, {features[0].shape}')       
        reshaped_features = [
            f.view(B, D, f.shape[1], f.shape[2]).contiguous() for f in features
        ]
        output = self.decoder(x.view(B, D, C, H, W).contiguous(), reshaped_features)
        return output   
        
class OpenClipUNETR(nn.Module):
    def __init__(self, embed_dim = 768, patch_size = 16, input_dim = 3, output_dim=3):
        super(OpenClipUNETR, self).__init__()
        
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.encoder = OpenClipEncoder()
        self.decoder = UNETRDecoder3D(embed_dim=self.embed_dim, patch_size=self.patch_size, input_dim=self.input_dim, output_dim=self.output_dim)
        
        # Freeze encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
            
    def forward(self, x):

        B, D, C, H, W = x.shape
        x = x.view(B * D, C, H, W).contiguous()
        features = self.encoder(x)        
        # print(f'encoder_features : {len(features)}, {features[0].shape}')       
        reshaped_features = [
            f.view(B, D, f.shape[1], f.shape[2]).contiguous() for f in features
        ]
        output = self.decoder(x.view(B, D, C, H, W).contiguous(), reshaped_features)
        return output

class SAMUNETR(nn.Module) :
    def __init__(self, embed_dim = 768, patch_size = 16, input_dim = 3, output_dim=3):
        super(SAMUNETR, self).__init__()

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.encoder = SamEncoder() 
        self.decoder = SAMDecoder3D(embed_dim=self.embed_dim, patch_size=self.patch_size, input_dim=self.input_dim, output_dim=self.output_dim)

        # Freeze encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):

        B, D, C, H, W = x.shape
        x = x.view(B * D, C, H, W).contiguous()
        features = self.encoder(x)        
        print(f'encoder_features : {len(features)}, {features[0].shape}')       
        reshaped_features = [
            f.view(B, D, f.shape[1], f.shape[2], f.shape[3]).contiguous() for f in features
        ]
        output = self.decoder(x.view(B, D, C, H, W).contiguous(), reshaped_features)
        return output


#test

if __name__ == '__main__':
    model = DINOUNETR()
    # x = torch.randn( 1, 4, 3, 1024,1024 ) #B, D, C, H, W
    x = torch.randn( 1, 4, 3, 224,224 ) #B, D, C, H, W
    output = model(x)
    
    print(f"UNETR output_shape : {output.shape}")