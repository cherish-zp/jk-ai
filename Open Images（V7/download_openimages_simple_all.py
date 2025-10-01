#!/usr/bin/env python3
"""
Open Images 数据下载器 (简化版)
直接从Google Cloud Storage下载Open Images V6数据集
"""

import os
import json
import requests
import pandas as pd
from PIL import Image
from tqdm import tqdm
import time
import random

class SimpleOpenImagesDownloader:
    def __init__(self, output_dir="/Users/zhangpeng/code_bigmodel/jk-ai/04-HuggingFace/data/open_image_v7_all"):
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, "images")
        self.annotations_dir = os.path.join(output_dir, "annotations")
        
        # 创建目录
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.annotations_dir, exist_ok=True)
        
        # Open Images V6 数据URL <mcreference link="https://medium.com/@Deeple.Mass/how-to-download-a-subset-of-open-image-dataset-v6-on-ubuntu-using-the-shell-c55336e33b03" index="1">1</mcreference>
        self.urls = {
            'validation_images': 'https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv',
            'validation_annotations': 'https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv',
            'class_descriptions': 'https://storage.googleapis.com/openimages/v6/oidv6-class-descriptions.csv'
        }
    
    def download_metadata(self):
        """下载元数据文件"""
        print("📥 下载元数据文件...")
        
        downloaded_files = {}
        
        for name, url in self.urls.items():
            try:
                print(f"  正在下载 {name}...")
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                
                file_path = os.path.join(self.annotations_dir, f"{name}.csv")
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                
                downloaded_files[name] = file_path
                print(f"  ✅ {name} 下载完成")
                
            except Exception as e:
                print(f"  ❌ {name} 下载失败: {e}")
                return None
        
        return downloaded_files
    
    def process_data(self, metadata_files, max_images=30):
        """处理数据并选择要下载的图片"""
        print("📊 处理数据...")
        
        try:
            # 读取图片列表
            images_df = pd.read_csv(metadata_files['validation_images'])
            print(f"📋 找到 {len(images_df)} 张验证集图片")
            
            # 读取标注
            annotations_df = pd.read_csv(metadata_files['validation_annotations'])
            print(f"🏷️ 找到 {len(annotations_df)} 个标注")
            
            # 读取类别描述
            class_desc_df = pd.read_csv(metadata_files['class_descriptions'], header=None, names=['LabelName', 'DisplayName'])
            class_dict = dict(zip(class_desc_df['LabelName'], class_desc_df['DisplayName']))
            
            # 合并数据
            merged_df = annotations_df.merge(images_df, on='ImageID', how='inner')
            
            # 随机选择图片
            if len(merged_df) > max_images:
                selected_images = merged_df['ImageID'].unique()[:max_images]
                merged_df = merged_df[merged_df['ImageID'].isin(selected_images)]
            
            # 添加类别描述
            merged_df['Description'] = merged_df['LabelName'].map(class_dict)
            
            print(f"🎯 选择了 {len(merged_df['ImageID'].unique())} 张图片进行下载")
            return merged_df, class_dict
            
        except Exception as e:
            print(f"❌ 处理数据失败: {e}")
            return None, None
    
    def download_image(self, image_url, image_id, max_retries=3):
        """下载单张图片"""
        for attempt in range(max_retries):
            try:
                response = requests.get(image_url, timeout=30)
                response.raise_for_status()
                
                # 保存图片
                image_path = os.path.join(self.images_dir, f"{image_id}.jpg")
                with open(image_path, 'wb') as f:
                    f.write(response.content)
                
                # 验证图片
                try:
                    with Image.open(image_path) as img:
                        width, height = img.size
                        if width < 100 or height < 100:
                            os.remove(image_path)
                            return None, "图片尺寸太小"
                        return image_path, None
                except Exception as e:
                    if os.path.exists(image_path):
                        os.remove(image_path)
                    return None, f"图片验证失败: {e}"
                    
            except Exception as e:
                if attempt == max_retries - 1:
                    return None, f"下载失败: {e}"
                time.sleep(random.uniform(1, 3))
        
        return None, "下载失败"
    
    def create_descriptions(self, data_df):
        """创建图片描述"""
        descriptions = {}
        
        # 按图片ID分组
        grouped = data_df.groupby('ImageID')
        
        for image_id, group in grouped:
            labels = group['Description'].dropna().tolist()
            confidence_scores = group['Confidence'].tolist()
            bboxes = []
            
            for _, row in group.iterrows():
                bbox = {
                    'label': row['Description'],
                    'confidence': row['Confidence'],
                    'xmin': row['XMin'],
                    'ymin': row['YMin'],
                    'xmax': row['XMax'],
                    'ymax': row['YMax']
                }
                bboxes.append(bbox)
            
            description = {
                'image_id': image_id,
                'original_url': group.iloc[0]['OriginalURL'],
                'labels': labels,
                'confidence_scores': confidence_scores,
                'bounding_boxes': bboxes,
                'description_text': f"这张图片包含: {', '.join(set(labels[:5]))}",
                'metadata': {
                    'license': group.iloc[0].get('License', ''),
                    'author': group.iloc[0].get('Author', ''),
                    'title': group.iloc[0].get('Title', '')
                }
            }
            descriptions[image_id] = description
        
        return descriptions
    
    def download_batch(self, max_images=30):
        """批量下载图片和描述"""
        print(f"🚀 开始下载 Open Images 数据集 (最多 {max_images} 张图片)")
        
        # 1. 下载元数据
        metadata_files = self.download_metadata()
        if not metadata_files:
            print("❌ 元数据下载失败")
            return
        
        # 2. 处理数据
        data_df, class_dict = self.process_data(metadata_files, max_images)
        if data_df is None:
            return
        
        # 3. 创建描述
        descriptions = self.create_descriptions(data_df)
        
        # 4. 下载图片
        print("📸 开始下载图片...")
        successful_downloads = []
        failed_downloads = []
        
        unique_images = data_df['ImageID'].unique()
        
        for image_id in tqdm(unique_images, desc="下载图片"):
            if image_id in descriptions:
                image_url = descriptions[image_id]['original_url']
                if image_url:
                    image_path, error = self.download_image(image_url, image_id)
                    if image_path:
                        successful_downloads.append(image_id)
                        
                        # 保存描述文件
                        desc_file = os.path.join(self.annotations_dir, f"{image_id}_description.json")
                        with open(desc_file, 'w', encoding='utf-8') as f:
                            json.dump(descriptions[image_id], f, ensure_ascii=False, indent=2)
                    else:
                        failed_downloads.append((image_id, error))
            
            # 添加延迟
            time.sleep(random.uniform(0.5, 1.0))
        
        # 5. 保存汇总信息
        summary = {
            'total_requested': len(unique_images),
            'successful_downloads': len(successful_downloads),
            'failed_downloads': len(failed_downloads),
            'success_rate': len(successful_downloads) / len(unique_images) * 100 if unique_images.size > 0 else 0,
            'downloaded_images': successful_downloads,
            'failed_images': failed_downloads,
            'dataset_info': {
                'source': 'Open Images V6',
                'split': 'validation',
                'total_classes': len(class_dict) if class_dict else 0
            }
        }
        
        summary_file = os.path.join(self.annotations_dir, "download_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ 下载完成!")
        print(f"📊 成功下载: {len(successful_downloads)} 张图片")
        print(f"❌ 下载失败: {len(failed_downloads)} 张图片")
        print(f"📈 成功率: {summary['success_rate']:.1f}%")
        print(f"📁 图片保存在: {self.images_dir}")
        print(f"📄 描述保存在: {self.annotations_dir}")

def main():
    """主函数"""
    print("🎨 Open Images 高质量图片下载器 (简化版)")
    print("=" * 50)
    
    # 创建下载器
    downloader = SimpleOpenImagesDownloader()
    
    # 开始下载
    downloader.download_batch(max_images=30)

if __name__ == "__main__":
    main()