import os


def rename_files(directory):
    for filename in os.listdir(directory):
        # 跳过已经符合命名格式的文件
        if filename.endswith('_enhanced.wav'):
            continue

        # 分离文件名和扩展名
        name, ext = os.path.splitext(filename)
        # 生成新的文件名
        new_name = f"{name}_enhanced{ext}"
        # 构建完整的文件路径
        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_name)

        # 重命名文件
        os.rename(old_path, new_path)
        print(f"Renamed: {old_path} to {new_path}")


if __name__ == "__main__":
    enhanced_dir = "C:\\Users\\sagacious h\\Pycharmprojects\\MANNER\\enhanced_1"
    rename_files(enhanced_dir)