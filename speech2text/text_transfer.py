import os

def merge_txt_files(input_dir, output_file):
    # 打开输出文件
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # 遍历目录下的所有文件
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    # 读取每个 .txt 文件的内容并写入输出文件
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        outfile.write(infile.read())
                        outfile.write("\n\n")  # 添加换行分隔符

# 示例用法
input_directory = r'E:\manner_datasets\DS_10283_2791\testset_txt\testset_txt'  # 替换为你的目录路径
output_file_path = 'merged_output.txt'   # 输出文件路径
merge_txt_files(input_directory, output_file_path)