## Encoders
**1. Dino v2**
 - model name : facebook/dinov2-base
 - patch size : 14x14
 - resolution : bxcx224x224
 - feature shape : bx256x768
   
**2. BEiT v2**
  - model name : microsoft/beit-base-patch16-224-pt22k-ft22k
  - patch size : 16x16
  - resolution : bxcx224x224
  - feature shape : bx197x768
    
**3. BLIP**
  - model name : Salesforce/blip-image-captioning-base
  - patch size : 16x16
  - resolution : bxcx224x224
  - feature shape : bx197x768
    
**4. OpenCLIP**
  - model name : openai/clip-vit-base-patch16
  - patch size : 16x16
  - resolution : bxcx224x224
  - feature shape : bx197x768
    
**5. SAM**
 - model name : openai/clip-vit-base-patch16
- patch size : 16x16
- resolution : bx3x1024x1024
- feature shape : bx4096x768

**6.MedSAM** 
- model name :
- patch size :
- resolution : bxcx256x256
- feature shape :


## Decoders

  **1. UNETRDecoder14** 
  - patch size 14 (Dino v2)

  **2. UNETRDecoder16**
  - patch size 16 (BEiT v2, BLIP, OPenCLIP)

**3. SAMDecoder**

