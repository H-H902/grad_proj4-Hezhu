import numpy as np
import torch
import librosa

import utils  # 包含 STFT 的工具模块

# 加载预训练模型
model = torch.jit.load('runs/BVCC/model_best.pt', map_location=torch.device('cpu'))

# 替换为目标音频路径
audio_path = r'C:\Users\sagacious h\Pycharmprojects\MANNER\resampled_28spk\noisy_test\p232_003.wav'

# 加载目标音频
samples, _ = librosa.load(audio_path, sr=16000)

# 计算 STFT
spec = torch.FloatTensor(utils.stft(samples))

# 推理
with torch.no_grad():
    prediction = model(spec[None, None, ...])

# 获取均值和方差
mean = prediction[:, 0]
variance = prediction[:, 1]
print(f'{mean=}, {variance=}')