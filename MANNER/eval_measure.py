import os
import json
import librosa
from tqdm import tqdm
import argparse
from src.metric import eval_composite
from os.path import join as opj


def eval_all_measure(clean_path, enhanced_path):
    clean_list = os.listdir(clean_path)
    enhanced_list = [i for i in os.listdir(enhanced_path) if '_enhanced.wav' in i]
    clean_list.sort()
    enhanced_list.sort()

    print(f'Clean path: {clean_path}')
    print(f'Enhanced path: {enhanced_path}')
    print(f'Found {len(clean_list)} clean files and {len(enhanced_list)} enhanced files')

    csig, cbak, covl, pesq_score, count = 0, 0, 0, 0, 0
    for clean_file, enhanced_file in tqdm(zip(clean_list, enhanced_list)):
        assert clean_file[:-4] in enhanced_file, 'Not matched clean and enhanced wav'

        clean = librosa.load(opj(clean_path, clean_file), sr=None)[0]
        enhance = librosa.load(opj(enhanced_path, enhanced_file), sr=None)[0]
        res = eval_composite(clean, enhance)
        csig += res['csig']
        cbak += res['cbak']
        covl += res['covl']
        pesq_score += res['pesq']
        count += 1

    print(f'CSIG: {csig / count}, CBAK: {cbak / count}, COVL: {covl / count}, PESQ: {pesq_score / count}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_path', type=str, required=True, help='Path to the clean test data')
    parser.add_argument('--enhanced_path', type=str, required=True, help='Path to the enhanced data')

    args = parser.parse_args()
    eval_all_measure(args.clean_path, args.enhanced_path)