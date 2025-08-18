import os
import argparse
import subprocess

def extract_audio_to_wav(source_dir, target_dir):
    """
    从 MP4 提取无损 WAV 音频（PCM 编码）
    :param source_dir: 原始视频目录
    :param target_dir: 目标音频目录
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for filename in os.listdir(source_dir):
        if filename.lower().endswith('.mp4'):
            mp4_path = os.path.join(source_dir, filename)
            wav_filename = os.path.splitext(filename)[0] + '.wav'
            wav_path = os.path.join(target_dir, wav_filename)

            try:
                # 使用 FFmpeg 提取无损 WAV（PCM 16-bit 立体声，44.1kHz）
                subprocess.run([
                    'ffmpeg',
                    '-i', mp4_path,    # 输入文件
                    '-vn',             # 禁用视频流
                    '-acodec', 'pcm_s16le',  # PCM 16-bit 无损编码
                    '-ar', '44100',   # 采样率（CD 标准）
                    '-ac', '2',        # 立体声
                    '-y',             # 覆盖输出文件（如果存在）
                    wav_path
                ], check=True)
                print(f"✓ 成功提取: {filename} -> {wav_filename}")
            except subprocess.CalledProcessError as e:
                print(f"✗ 转换失败: {filename} (错误: {e})")


if __name__ == "__main__":

    # 直接硬编码路径
    source_path = "/Users/zhangpeng/Downloads/合集作品/声娱文化《剑来》第5季/sp_1_mp4/"
    target_dir = "/Users/zhangpeng/Downloads/合集作品/声娱文化《剑来》第5季/sp_1_mp3/"

    # 调用函数
    extract_audio_to_wav(source_path, target_dir)