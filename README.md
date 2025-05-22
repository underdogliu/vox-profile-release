## Vox-Profile: A Speech Foundation Model Benchmark for Characterizing Diverse Speaker and Speech Traits

#### In this repo, we present Vox-Profile [[Paper Link](https://arxiv.org/pdf/2505.14648)], one of the first benchmarking efforts that systematically evaluate rich multi-dimensional speaker and speech traits from English-speaking voices. Our benchmark is presented below:

<div align="center">
 <img src="img/vox-profile.png" width="800px">
</div>

### Our training data filters output audio shorter than 3 seconds (unreliable predictions) and longer than 15 seconds (computation limitation), so you need to cut your audio to a maximum of 15 seconds, 16kHz and mono channel

### Download Repo
```
git clone git@github.com:tiantiaf0627/vox-profile-release.git
```

### Installation
```
conda create -n vox_profile python=3.8
cd vox-profile-release
pip install -e .
```

### Quick Example - WavLM Large Narrow Accent
```
# Load libraries
import torch
import torch.nn.functional as F
from src.model.accent.wavlm_accent import WavLMWrapper

# Label List
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
    
# Probability and output
wavlm_prob = F.softmax(wavlm_logits, dim=1)
print(english_accent_list[torch.argmax(wavlm_prob).detach().cpu().item()])
```

#### Given that the Vox-Profile Benchmark paper is still under peer-review, we provide limited set of models and model weights before the review is concluded. But below are the models we currently put out.

### WavLM-Large Models

 Model Name  | Data  | Pre-trained Model | Use LoRa |  LoRa Rank Size  | Output | Example Code |
|--------------------------------------------------------|-------|-----------------|-----------------|-------------|------------------------|------------------------|
| [wavlm-large-sex-age](https://huggingface.co/tiantiaf/wavlm-large-age-sex)   | CommonVoice+Timit+Voxceleb (age enriched) | wavlm-large              | Yes              | 16              | Sex (2-class) / Age (0-1)*100 Years  | [Example](https://github.com/tiantiaf0627/vox-profile-release/blob/main/src/example/age_sex_wavlm.py) |
| [wavlm-large-broader-accent](https://huggingface.co/tiantiaf/wavlm-large-broader-accent)   | See Paper (11 Datasets) | wavlm-large              | Yes              | 16              | North American / British / Other (3-class)  | [Example](https://github.com/tiantiaf0627/vox-profile-release/blob/main/src/example/broad_accent_wavlm.py) |
| [wavlm-large-narrow-accent](https://huggingface.co/tiantiaf/wavlm-large-narrow-accent)   | See Paper (11 Datasets) | wavlm-large              | Yes              | 16              | See example  | [Example](https://github.com/tiantiaf0627/vox-profile-release/blob/main/src/example/narrow_accent_wavlm.py) |
| [wavlm-large-voice-quality](https://huggingface.co/tiantiaf/wavlm-large-voice-quality)   | ParaSpeechCaps | wavlm-large              | Yes              | 16              | See example  | [Example](https://github.com/tiantiaf0627/vox-profile-release/blob/main/src/example/voice_quality_wavlm.py) |
| [wavlm-large-influency](https://huggingface.co/tiantiaf/wavlm-large-speech-flow)   | SEP28K+FluencyBank | wavlm-large              | Yes              | 16              | Fluent/Disfluent (Specified Disfluency Types)  | [Example](https://github.com/tiantiaf0627/vox-profile-release/blob/main/src/example/fluency_wavlm.py) |
| [wavlm-large-categorical-emotion](https://huggingface.co/tiantiaf/wavlm-large-categorical-emotion)   | MSP-Podcast | wavlm-large              | No              | NA              | 8 Emotions + Other  | [Example](https://github.com/tiantiaf0627/vox-profile-release/blob/main/src/example/categorized_emotion_wavlm.py) |


### Whisper-Large V3 Models
Model Name  | Data  | Pre-trained Model | Use LoRa |  LoRa Rank Size  | Output | Example Code |
|--------------------------------------------------------|-------|-----------------|-----------------|-------------|------------------------|------------------------|
| [whisper-large-v3-broader-accent](https://huggingface.co/tiantiaf/whisper-large-v3-broad-accent)   | See Paper (11 Datasets) | whisper-large v3              | Yes              | 16              | North American / British / Other (3-class)  | [Example](https://github.com/tiantiaf0627/vox-profile-release/blob/main/src/example/broad_accent_whisper.py) |
| [whisper-large-v3-narrow-accent](https://huggingface.co/tiantiaf/whisper-large-v3-narrow-accent)   | See Paper (11 Datasets) | whisper-large v3             | Yes              | 16              | See example  | [Example](https://github.com/tiantiaf0627/vox-profile-release/blob/main/src/example/narrow_accent_whisper.py) |
| [whisper-large-v3-voice-quality](https://huggingface.co/tiantiaf/whisper-large-v3-voice-quality)   | ParaSpeechCaps | wavlm-large              | Yes              | 16              | See example  | [Example](https://github.com/tiantiaf0627/vox-profile-release/blob/main/src/example/voice_quality_whisper.py) |
| [whisper-large-v3-influency](https://huggingface.co/tiantiaf/whisper-large-v3-speech-flow)   | SEP28K+FluencyBank | wavlm-large              | Yes              | 16              | Fluent/Disfluent (Specified Disfluency Types)  | [Example](https://github.com/tiantiaf0627/vox-profile-release/blob/main/src/example/fluency_whisper.py) |
| [whisper-large-v3-categorical-emotion](https://huggingface.co/tiantiaf/whisper-large-v3-msp-podcast-emotion)   | MSP-Podcast | whisper-large v3             | Yes              | 16              | 8 Emotions + Other  | [Example](https://github.com/tiantiaf0627/vox-profile-release/blob/main/src/example/categorized_emotion_whisper.py) |


### Labeling Scheme
In Vox-Profile, we experiments with over 15 publicly available datasets to predict static traits (speaker age, speaker sex, speaker accent, and voice quality) and dynamic traits (speech emotion, speech flow, and speech expressiveness) in different recording conditions and elicitation settings (e.g., read, spontaneous, and conversational speech). Our labeling taxonomy is described below:

<div align="center">
 <img src="img/label_mapping.png" width="400px">
</div>

### Enabling Versatile Speech Applications with Vox-Profile

Our Vox-Profile can be used as a fundamental component to support a versatile speech applications.

#### 1. Speech Model Performance Analysis

We generate speaker and speech traits for existing datasets and investigate whether these generated labels can lead to the same insights as using the ground truth trait information in analyzing the speech model performances. For example, in the results below, we identify that regardless of grouping by groud truth traits or traits inferred by Vox-Profile, the ASR performance trends remain the same across different traits.

<div align="center">
 <img src="img/wer.png" width="800px">
</div>

#### 2. Automated Evaluation Tool for Speech Generation Tasks

We demonstrate the utility of Vox-Profile as an evaluation tool for speech generation tasks by comparing two representative models: FreeVC and VALLE-X. As shown in Table below, the accent prediction scores and the cosine similarity for the synthesized samples from FreeVC suggest greater similarity to the source speaker’s accent than to the reference speaker. In contrast, the scores for VALLE-X indicate closer alignment with the reference speaker’s accent in most conditions. These findings are consistent with previous studies, which report that FreeVC has limited capability in replicating the accentual features of the reference speaker compare to VALLE-X.

<div align="center">
 <img src="img/vc_evaluation.png" width="800px">
</div>


#### 3. Generating Synthetic Speaking Style Prompt
Vox-Profile provides a more extensive and varied set of traits, including speech flow, arousal, valence, and speaker age. Moreover, computational models of Vox-Profile output probabilistic predictions for each trait, enabling more nuanced and confidence sensitive descriptions. For example, a Scottish accent prediction with a probability of 0.9 can be described as having a distinct Scottish accent.

<div align="center">
 <img src="img/speaking_prompt.png" width="800px">
</div>

Human-evaluation results comparing synthetic speaking style prompts from Vox-Profile and human-annotated speaking style prompts from ParaSpeechCaps suggest that this group of human raters shows similar preference levels for both synthetic and human-annotated speaking style prompts. Specifically, they favor the emotion, age, and speech flow descriptions generated by Vox-Profile over those from ParaSpeechCaps.
