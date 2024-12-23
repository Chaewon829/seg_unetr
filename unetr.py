import UNET.seg_unet.decoder as decoder
from UNET.seg_unet.decoder import UNETRDecoder14, UNETRDecoder16, DummyEncoder, SAMDecoder
from UNET.seg_unet.encoder import Dinov2Encoder, BLIPEncoder, OpenClipEncoder ,BEiTv2Encoder, SamEncoder
import sys
import os
import torch
import torch.nn as nn

encoders = {
    'dinov2': Dinov2Encoder,
    'blip': BLIPEncoder,
    'openclip': OpenClipEncoder
}

decoder = {
    'unetr1': UNETRDecoder1
}

class TESTUNETR(nn.Module):
    def __init__(self, embed_dim=768, patch_size = 16, input_dim=3, output_dim=3):
        super().__init__()
        self.encoder = DummyEncoder()
        self.decoder = UNETRDecoder1(embed_dim,patch_size, input_dim, output_dim)
        # Freeze encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
    def forward(self, x):
        features = self.encoder(x)
        print(f'encoder_features : {len(features)}, {features[0].shape}')
        output = self.decoder(x, features)
        return output
        
    
class DINOUNETR(nn.Module):
    def __init__(self, embed_dim = 768, patch_size = 14, input_dim = 3, output_dim=3):
        super(DINOUNETR, self).__init__()
        
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.encoder = Dinov2Encoder()
        self.decoder = UNETRDecoder14(embed_dim=self.embed_dim, patch_size=self.patch_size, input_dim=self.input_dim, output_dim=self.output_dim)
        
        # Freeze encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        features = self.encoder(x)
        print(f'encoder_features : {len(features)}, {features[0].shape}')
        output = self.decoder(x, features)
        return output
    
class BeiTUNETR(nn.Module):
    def __init__(self, embed_dim = 768, patch_size = 16, input_dim = 3, output_dim=3):
        super(BeiTUNETR, self).__init__()
        
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.encoder = BEiTv2Encoder()
        self.decoder = UNETRDecoder16(embed_dim=self.embed_dim, patch_size=self.patch_size, input_dim=self.input_dim, output_dim=self.output_dim) 

        # Freeze encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        features = self.encoder(x)
        print(f'encoder_features : {len(features)}, {features[0].shape}')
        output = self.decoder(x, features)
        return output
    
    
class BLIPUNETR(nn.Module):
    def __init__(self, embed_dim = 768, patch_size = 16, input_dim = 3, output_dim=3):
        super(BLIPUNETR, self).__init__()
        
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.encoder = BLIPEncoder()
        self.decoder = UNETRDecoder16(embed_dim=self.embed_dim, patch_size=self.patch_size, input_dim=self.input_dim, output_dim=self.output_dim)

        # Freeze encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        features = self.encoder(x)
        print(f'encoder_features : {len(features)}, {features[0].shape}')
        output = self.decoder(x, features)
        return output
    
    
class OpenClipUNETR(nn.Module):
    def __init__(self, embed_dim = 768, patch_size = 16, input_dim = 3, output_dim=3):
        super(OpenClipUNETR, self).__init__()
        
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.encoder = OpenClipEncoder()
        self.decoder = UNETRDecoder16(embed_dim=self.embed_dim, patch_size=self.patch_size, input_dim=self.input_dim, output_dim=self.output_dim)
        
        # Freeze encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        features = self.encoder(x)
        print(f'encoder_features : {len(features)}, {features[0].shape}')
        output = self.decoder(x, features)
        return output


class SAMUNETR(nn.Module) :
    def __init__(self, embed_dim = 768, patch_size = 16, input_dim = 3, output_dim=3):
        super(SAMUNETR, self).__init__()

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.encoder = SamEncoder() 
        self.decoder = SAMDecoder(embed_dim=self.embed_dim, patch_size=self.patch_size, input_dim=self.input_dim, output_dim=self.output_dim)

        # Freeze encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x) : 
        features = self.encoder(x)
        print(f'encoder_features : {len(features)}, {features[0].shape}')
        output = self.decoder(x, features)
        return output
    

# class MedSAMUNETR(nn.Module) :
#     pass 




#test

if __name__ == '__main__':
    model = SAMUNETR()
    print(model)
    x = torch.randn(1, 3, 1024, 1024)
    output = model(x)
    print(f"output_shape : {output.shape}")
    
    
    
