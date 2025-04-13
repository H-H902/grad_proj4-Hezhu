import warnings
warnings.filterwarnings('ignore')

import sys
import os
import json
import yaml
import torch
import argparse
from tqdm import tqdm

from src.dataset import *
from src.utils import *
from src.models import MANNER as MANNER_BASE
from src.models_small import MANNER as MANNER_SMALL 

def main(args):
    
    seed_init()
    
    # Select model version
    if 'base' in args.model_name:
        model = MANNER_BASE(in_channels=1, out_channels=1, hidden=60, depth=4, kernel_size=8, stride=4, growth=2, head=1, segment_len=64).to(args.device)
    elif 'large' in args.model_name:
        model = MANNER_BASE(in_channels=1, out_channels=1, hidden=120, depth=4, kernel_size=8, stride=4, growth=2, head=1, segment_len=64).to(args.device)
    elif 'small' in args.model_name:
        model = MANNER_SMALL(in_channels=1, out_channels=1, hidden=60, depth=4, kernel_size=8, stride=4, growth=2, head=1, segment_len=64).to(args.device)
        
    checkpoint = torch.load(f'./weights/{args.model_name}')
    model.load_state_dict(checkpoint['state_dict'])
    print(f'--- Load {args.model_name} weights ---')
        
    model.eval()
    with torch.no_grad():
        # with open(args.input_test_file, 'r', encoding='utf-8') as fi:
            # file_list = [x.split('|')[0] for x in fi.read().split('\n') if len(x) > 0]
        file_list = os.listdir(args.noisy_path)
        output_path = args.esti_file_path # you can change the output path
        noisy_list  = file_list
        for n_file in tqdm(noisy_list):
            
            noisy, sr = torchaudio.load(f'{args.noisy_path}/{n_file}')
            if sr != 16000:
                tf    = torchaudio.transforms.Resample(sr, 16000)
                noisy = tf(noisy)
            
            noisy    = noisy.unsqueeze(0).to(args.device)
            enhanced = model(noisy)
            enhanced = enhanced.squeeze(0).detach().cpu()
            os.makedirs(os.path.join(args.esti_file_path), exist_ok=True)
            # save_name = n_file.split('.')[0] + '_enhanced.wav' 
            torchaudio.save(f'{output_path}/{n_file}', enhanced, 16000)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0', help='Cuda device')
    parser.add_argument('--noisy_path', type=str, default="C:\\Users\\sagacious h\\Pycharmprojects\\MANNER\\test_noise", help='Noisy input folder')
    parser.add_argument('--esti_file_path', type=str, default="C:\\Users\\sagacious h\\Pycharmprojects\\MANNER\\enhanced_1")
    parser.add_argument('--input_test_file', default=r'C:\Users\sagacious h\Pycharmprojects\MANNER\testset_txt\testset_txt')
    parser.add_argument('--model_name', type=str, default='manner_base .pth', help='Model name')
    
    args = parser.parse_args()
    main(args)
