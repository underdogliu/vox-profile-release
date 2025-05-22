import torch
import torchaudio
import torch.nn.functional as F
from src.model.emotion.whisper_emotion import WhisperWrapper

emotion_list = [
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

# Load model from Huggingface
model = WhisperWrapper.from_pretrained("tiantiaf/whisper-large-v3-msp-podcast-emotion").to(device)
model.eval()

# Load data, here just zeros as the example, audio data should be 16kHz mono channel
data, _ = torchaudio.load("YOUR_DATA")
data = data.float().to(device)
logits, embedding, _, _, _, _ = model(data, return_feature=True)
    
# Probability
emotion_prob = F.softmax(logits, dim=1)
print(emotion_list[torch.argmax(emotion_prob).detach().cpu().item()])