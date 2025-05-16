import torch
import sys, os, pdb
import argparse, logging
import torch.nn.functional as F

from pathlib import Path


sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1])))
sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1]), 'model', 'accent'))

from wavlm_dialect import WavLMWrapper
from whisper_dialect import WhisperWrapper


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


def parse_finetune_args():
    # parser
    parser = argparse.ArgumentParser(description='finetune experiments')
    parser.add_argument(
        '--num_layers',
        default=1,
        type=int,
        help="num of layers",
    )

    parser.add_argument(
        '--hidden_size',
        default=256,
        type=int,
        help="hidden size",
    )
    parser.add_argument(
        '--finetune_method', 
        default='finetune',
        type=str, 
        help='finetune method: adapter, embedding prompt, lora'
    )
    
    parser.add_argument(
        '--lora_rank', 
        default=16,
        type=int, 
        help='lora rank'
    )

    parser.add_argument(
        '--dataset_rev', 
        action='store_true',
        help='reverse dataset'
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    label_list = [
        'British Isles', 'North America', 'Other'
    ]
    
    # Find device
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available(): print('GPU available, use GPU')

    # Define the model
    # Note that ensemble yields the better performance than the single model
    model_path = "model"
    # Define the model wrapper
    wavlm_model = model = WavLMWrapper(
        pretrain_model="wavlm_large", 
        finetune_method="lora",
        lora_rank=16,
        output_class_num=3,
        freeze_params=False, 
        use_conv_output=True,
        apply_gradient_reversal=False, 
        num_dataset=3
    ).to(device)
    
    whisper_model = WhisperWrapper(
        pretrain_model="whisper_large", 
        finetune_method="lora",
        lora_rank=16,
        output_class_num=3,
        freeze_params=False, 
        use_conv_output=True,
        apply_gradient_reversal=False, 
        num_dataset=11
    ).to(device)
        
    wavlm_model.load_state_dict(torch.load(os.path.join(model_path, f"wavlm_us_uk.pt"), weights_only=True), strict=False)
    wavlm_model.load_state_dict(torch.load(os.path.join(model_path, f"wavlm_us_uk_lora.pt")), strict=False)
    
    whisper_model.load_state_dict(torch.load(os.path.join(model_path, f"whisper_us_uk.pt"), weights_only=True), strict=False)
    whisper_model.load_state_dict(torch.load(os.path.join(model_path, f"whisper_us_uk_lora.pt")), strict=False)
    
    wavlm_model.eval()
    whisper_model.eval()
    
    data = torch.zeros([1, 16000]).to(device)
    wavlm_logits, wavlm_embeddings      = wavlm_model(data, return_feature=True)
    whisper_logits, whisper_embeddings  = whisper_model(data, return_feature=True)

    ensemble_logits = (wavlm_logits + whisper_logits) / 2
    ensemble_prob   = F.softmax(ensemble_logits, dim=1)
    
    print(ensemble_prob.shape)
    print(wavlm_embeddings.shape)

