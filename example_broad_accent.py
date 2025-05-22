import torch
import torch.nn.functional as F
from src.model.accent.wavlm_accent import WavLMWrapper

english_accent_list = [
    'East Asia', 'English', 'Germanic', 'Irish', 
    'North America', 'Northern Irish', 'Oceania', 
    'Other', 'Romance', 'Scottish', 'Semitic', 'Slavic', 
    'South African', 'Southeast Asia', 'South Asia', 'Welsh'
]
    
# Find device
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# Load model from Huggingface
wavlm_model = WavLMWrapper.from_pretrained("tiantiaf/wavlm-large-narrow-accent").to(device)
wavlm_model.eval()

# Load data, here just zeros as the example, audio data should be 16kHz mono channel
data = torch.zeros([1, 16000]).float().to(device)
wavlm_logits, wavlm_embeddings = wavlm_model(data, return_feature=True)
    
# Probability
wavlm_prob = F.softmax(wavlm_logits, dim=1)
accent_label = print(english_accent_list[torch.argmax(wavlm_prob).detach().cpu().item()])