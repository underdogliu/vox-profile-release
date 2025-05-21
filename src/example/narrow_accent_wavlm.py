import torch
import sys, os, pdb
import argparse, logging
import torch.nn.functional as F

from pathlib import Path


sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1])))
sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1]), 'model', 'accent'))

from wavlm_accent import WavLMWrapper

# define logging console
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-3s ==> %(message)s', 
    level=logging.INFO, 
    datefmt='%Y-%m-%d %H:%M:%S'
)

os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 


if __name__ == '__main__':

   
    label_list = [
        'East Asia', 'English', 'Germanic', 'Irish', 
        'North America', 'Northern Irish', 'Oceania', 
        'Other', 'Romance', 'Scottish', 'Semitic', 'Slavic', 
        'South African', 'Southeast Asia', 'South Asia', 'Welsh'
    ]
    
    # Find device
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available(): print('GPU available, use GPU')

    # Define the model
    wavlm_model = WavLMWrapper(
        pretrain_model="wavlm_large", 
        finetune_method="lora",
        lora_rank=16,
        output_class_num=16,
        freeze_params=True, 
        use_conv_output=True,
        apply_gradient_reversal=False, 
        num_dataset=3
    )
    
    wavlm_model = wavlm_model.from_pretrained("tiantiaf/wavlm-large-narrow-accent").to(device)
    wavlm_model.eval()
        
    data = torch.zeros([1, 16000]).float().to(device)
    wavlm_logits, wavlm_embeddings = wavlm_model(data, return_feature=True)
    
    # Probability
    wavlm_prob = F.softmax(wavlm_logits, dim=1)
    