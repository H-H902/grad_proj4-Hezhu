import argparse
import numpy as np


def load_file(file_path):
    """加载文本文件内容"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


def calculate_wer(reference, hypothesis):
    """计算词错误率(Word Error Rate, WER)"""
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    # 初始化编辑距离矩阵
    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1))
    for i in range(len(ref_words) + 1):
        d[i, 0] = i
    for j in range(len(hyp_words) + 1):
        d[0, j] = j

    # 计算编辑距离
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i, j] = d[i - 1, j - 1]
            else:
                substitution = d[i - 1, j - 1] + 1
                insertion = d[i, j - 1] + 1
                deletion = d[i - 1, j] + 1
                d[i, j] = min(substitution, insertion, deletion)

    wer = d[len(ref_words), len(hyp_words)] / len(ref_words)
    return wer


def calculate_cer(reference, hypothesis):
    """计算字错误率(Character Error Rate, CER)"""
    ref_chars = list(reference.replace(" ", ""))
    hyp_chars = list(hypothesis.replace(" ", ""))

    # 初始化编辑距离矩阵
    d = np.zeros((len(ref_chars) + 1, len(hyp_chars) + 1))
    for i in range(len(ref_chars) + 1):
        d[i, 0] = i
    for j in range(len(hyp_chars) + 1):
        d[0, j] = j

    # 计算编辑距离
    for i in range(1, len(ref_chars) + 1):
        for j in range(1, len(hyp_chars) + 1):
            if ref_chars[i - 1] == hyp_chars[j - 1]:
                d[i, j] = d[i - 1, j - 1]
            else:
                substitution = d[i - 1, j - 1] + 1
                insertion = d[i, j - 1] + 1
                deletion = d[i - 1, j] + 1
                d[i, j] = min(substitution, insertion, deletion)

    cer = d[len(ref_chars), len(hyp_chars)] / len(ref_chars)
    return cer


def main():
    parser = argparse.ArgumentParser(description="计算WER和CER")
    parser.add_argument("--ref", required=True, help="标准答案文件路径")
    parser.add_argument("--hyp", required=True, help="测试文本文件路径")
    args = parser.parse_args()

    # 加载文件
    ref_lines = load_file(args.ref)
    hyp_lines = load_file(args.hyp)

    if len(ref_lines) != len(hyp_lines):
        print(f"警告：文件行数不匹配（参考:{len(ref_lines)}，测试:{len(hyp_lines)}）")
        min_lines = min(len(ref_lines), len(hyp_lines))
        ref_lines = ref_lines[:min_lines]
        hyp_lines = hyp_lines[:min_lines]

    # 计算每行的WER和CER
    total_wer = 0.0
    total_cer = 0.0
    num_lines = len(ref_lines)

    print("行号\tWER\tCER\t参考文本")
    print("-" * 80)
    for i, (ref, hyp) in enumerate(zip(ref_lines, hyp_lines), 1):
        wer = calculate_wer(ref, hyp)
        cer = calculate_cer(ref, hyp)
        total_wer += wer
        total_cer += cer
        print(f"{i}\t{wer:.4f}\t{cer:.4f}\t{ref[:50]}...")

    # 计算平均值
    avg_wer = total_wer / num_lines
    avg_cer = total_cer / num_lines

    print("\n统计结果:")
    print(f"平均WER: {avg_wer:.4f}")
    print(f"平均CER: {avg_cer:.4f}")
    print(f"总行数: {num_lines}")


if __name__ == "__main__":
    main()