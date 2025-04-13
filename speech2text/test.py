import whisper
import os
from tqdm import tqdm  # 用于显示进度条

whisper_model = whisper.load_model("large.pt")
input_directory = r"C:\Users\sagacious h\Pycharmprojects\MANNER\resampled_28spk\noisy_test"  # 替换为你的目录路径
output_file = "output_noise.txt"  # 替换为你的输出文件路径

# 获取所有 .wav 文件
wav_files = [file_name for file_name in os.listdir(input_directory) if file_name.endswith(".wav")]

with open(output_file, "w", encoding="utf-8") as f:
    # 使用 tqdm 显示进度条
    for file_name in tqdm(wav_files, desc="Processing files"):
        file_path = os.path.join(input_directory, file_name)
        result = whisper_model.transcribe(file_path)
        text = ", ".join([i["text"] for i in result["segments"] if i is not None])
        f.write(f"\n{text}\n\n")