import torch
import logging
import torchaudio
import sys, os, pdb
import torch.nn.functional as F

from pathlib import Path

sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1])))
sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1]), 'model', 'emotion'))

from whisper_emotion import WhisperWrapper


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
        'Anger', 
        'Contempt', 
        'Disgust', 
        'Fear', 
        'Happiness', 
        'Neutral', 
        'Sadness', 
        'Surprise', 
        'Other'
    ]

    # Find device
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available(): print('GPU available, use GPU')

    # Define the model
    # Note that ensemble yields the better performance than the single model
    model = WhisperWrapper(
        pretrain_model="whisper_large", 
        finetune_method="lora",
        lora_rank=16, 
        output_class_num=9,
        freeze_params=True, 
        use_conv_output=True,
        detailed_class_num=17
    ).to(device)
        
    model = model.from_pretrained("tiantiaf/whisper-large-v3-msp-podcast-emotion").to(device)
    model.eval()
    
    # Audio must be 16k Hz
    data = torch.zeros([1, 16000]).float().to(device)
    logits, embedding, _, _, _, _ = model(
        data, return_feature=True
    )
    
    emotion_prob   = F.softmax(logits, dim=1)

    print(emotion_prob.shape)
    print(embedding.shape)
    print(label_list[torch.argmax(emotion_prob).detach().cpu().item()])
    