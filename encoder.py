import sys
import os
import torch
import torch.nn as nn
from transformers import AutoModel, BeitModel, BlipModel, CLIPModel, SamModel




class Dinov2Encoder(nn.Module):
    def __init__(self, model_name = 'facebook/dinov2-base'):
        super(Dinov2Encoder, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)

    def forward(self, x):
        outputs = []
        with torch.no_grad(): 
            x = self.model.embeddings.patch_embeddings(x) #patch로 나누고 embedding
            for i, layer in enumerate(self.model.encoder.layer):
                x = layer(x)[0] if isinstance(layer(x), tuple) else layer(x)
                if i+1 in [3, 6, 9, 12]:
                    outputs.append(x)
        return outputs

class BEiTv2Encoder(nn.Module):
    def __init__(self, model_name='microsoft/beit-base-patch16-224-pt22k-ft22k'):
        super(BEiTv2Encoder, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)

    
    def forward(self, x):
        outputs = []
        with torch.no_grad():
            batch_size, channels, height, width = x.size()
            resolution = (height, width)
            embedding_output = self.model.embeddings(x)[0]
            
            for i, layer in enumerate(self.model.encoder.layer):
                # 각 레이어 호출
                layer_outputs = layer(embedding_output, resolution=resolution)
                
                if isinstance(layer_outputs, tuple):
                    embedding_output = layer_outputs[0]
                else : 
                    embedding_output = layer_outputs
                if i + 1 in [3, 6, 9, 12]:
                    outputs.append(embedding_output)
        
        return outputs
        

class BLIPEncoder(nn.Module):
    def __init__(self, model_name='Salesforce/blip-image-captioning-base'):
        """
        BLIP Vision Model 초기화
        """
        super(BLIPEncoder, self).__init__()
        self.model = BlipModel.from_pretrained(model_name).vision_model
        self.encoder_layers = self.model.encoder.layers
        self.embeddings = self.model.embeddings
        self.target_layers = [2, 5, 8, 11]  # 0-index 기준

    def forward(self, x):
        outputs = []  # 선택된 레이어 출력을 저장할 리스트
        
        with torch.no_grad():
            # 이미지 입력을 Embedding 단계에 전달
            embeddings = self.embeddings(x)
            # embeddings = self.dropout(embeddings)
            
            # Attention Mask 생성
            batch_size, seq_len, _ = embeddings.size()
            print(f"batch_size, seq_len, _ : {batch_size, seq_len, _}")
            attention_mask = torch.ones(batch_size, 1, 1, seq_len).to(embeddings.device)

            
            # Vision Transformer Encoder 레이어를 순차적으로 통과
            for i, layer in enumerate(self.encoder_layers):
                outputs_layer = layer(embeddings, attention_mask=attention_mask)
                embeddings = outputs_layer[0] if isinstance(outputs_layer, tuple) else outputs_layer
                
                # 대상 레이어 출력 저장
                if i in self.target_layers:
                    outputs.append(embeddings)
        
        return outputs


class OpenClipEncoder(nn.Module):
    def __init__(self, model_name='openai/clip-vit-base-patch16'):
       super(OpenClipEncoder, self).__init__()
       self.model = CLIPModel.from_pretrained(model_name)
       self.vision_model = self.model.vision_model
       self.encoder_layers = self. vision_model.encoder.layers
       
    def forward(self, x) :
        selected_outputs = []
        with torch.no_grad():
            embeddings = self.vision_model.embeddings(x)
            
            batch_size, seq_len, _ = embeddings.size()
            attention_mask = torch.ones(batch_size, 1, seq_len, seq_len)
            causal_attention_mask = torch.zeros(batch_size,1, seq_len, seq_len)
            
            for i, layer in enumerate(self.encoder_layers):
                outputs = layer(embeddings, attention_mask = attention_mask, causal_attention_mask=causal_attention_mask)
                embeddings = outputs[0] if isinstance(outputs, tuple) else outputs
                if i+1 in [3, 6, 9, 12]:
                    selected_outputs.append(embeddings)
        return selected_outputs    
    
    
class SamEncoder(nn.Module):
    def __init__(self, model_name='facebook/sam-vit-base'):
        super(SamEncoder, self).__init__()
        self.model = SamModel.from_pretrained(model_name).vision_encoder
        
    def forward(self, x):
        outputs = []
        b, c, h, w = x.size()
        with torch.no_grad():
            # Patch Embedding
            x = self.model.patch_embed.projection(x)  # (B, Hidden_dim, H/16, W/16)
            x = x.permute(0, 2, 3, 1)  # (B, H/16, W/16, Hidden_dim)
            
            # Vision Transformer Layers
            for i, layer in enumerate(self.model.layers):
                x = layer(x)
                if isinstance(x, tuple):
                    x = x[0]  # 첫 번째 요소만 사용
                    b, h, w, emb_dim = x.size()
                
                if i + 1 in [3, 6, 9, 12]:
                    outputs.append(x.view(b, -1,emb_dim))
                
        
        return outputs 


# class MedSAMEncoder(nn.Module) :
#     pass
        
if __name__ == '__main__':

    dummy_image = torch.randn(1, 3, 1024, 1024)  # (batch_size, channels, height, width)
    encoder = SamEncoder()
    features = encoder(dummy_image)

    print(len(features))
    print(features[0].shape)
