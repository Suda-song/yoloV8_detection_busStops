#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
补充区域信息程序
用于查找final_station_ranking_with_district.csv中district字段缺失的条目，
重新发起API请求补充完整数据
"""

import requests
import pandas as pd
import time
import threading
import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from datetime import datetime
import sys
import numpy as np

# ==================== 配置区域 ====================
# 百度API配置
BAIDU_AK = "olQ5vPbNSFmY2xGL7mFZcIEbSQjbBCIh"  # 请替换为您的百度地图API Key
BASE_URL = "https://api.map.baidu.com/reverse_geocoding/v3/"

# 文件配置
INPUT_CSV = "/Users/songdingan/Downloads/detect_bus/final_station_ranking_with_district.csv"  # 输入文件
OUTPUT_CSV = "/Users/songdingan/Downloads/detect_bus/final_station_ranking_complete.csv"  # 输出文件
PROGRESS_CSV = "/Users/songdingan/Downloads/detect_bus/补充进度.csv"  # 进度保存文件

# 并发配置
MAX_WORKERS = 40  # 最大并发数
REQUEST_INTERVAL = 0.025  # 请求间隔（秒），40个并发约等于1000+ QPS，需要根据API限制调整

# API参数配置
API_PARAMS = {
    'extensions_poi': 1,
    'entire_poi': 1,
    'sort_strategy': 'distance',
    'output': 'json',
    'coordtype': 'bd09ll'
}
# ==================================================

class DistrictCompleter:
    """区域信息补充器"""
    
    def __init__(self, baidu_ak, input_csv, output_csv, progress_csv):
        self.baidu_ak = baidu_ak
        self.input_csv = input_csv
        self.output_csv = output_csv
        self.progress_csv = progress_csv
        
        # 线程安全控制
        self.write_lock = threading.Lock()
        self.request_lock = threading.Lock()
        self.last_request_time = 0
        
        # 数据存储
        self.data = None
        self.missing_records = []
        self.processed_records = set()
        self.updated_data = {}  # 存储更新的数据
        
        # 设置日志
        self.setup_logging()
        
    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('district_complete.log', encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_data(self):
        """加载输入数据"""
        try:
            self.logger.info(f"📂 正在加载数据: {self.input_csv}")
            
            if not Path(self.input_csv).exists():
                raise FileNotFoundError(f"输入文件不存在: {self.input_csv}")
            
            self.data = pd.read_csv(self.input_csv)
            
            # 验证必要的列
            required_cols = ['longitude', 'latitude']
            missing_cols = [col for col in required_cols if col not in self.data.columns]
            if missing_cols:
                raise ValueError(f"输入文件缺少必要的列: {missing_cols}")
            
            self.logger.info(f"✅ 数据加载成功!")
            self.logger.info(f"   - 样本数量: {len(self.data)}")
            self.logger.info(f"   - 列名: {list(self.data.columns)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 数据加载失败: {e}")
            return False
    
    def find_missing_records(self):
        """查找district字段缺失的记录"""
        try:
            self.logger.info("🔍 正在查找district字段缺失的记录...")
            
            # 检查district列是否存在
            if 'district' not in self.data.columns:
                # 如果没有district列，所有记录都需要处理
                self.missing_records = list(self.data.index)
                self.logger.info(f"   - 未找到district列，需要处理所有 {len(self.missing_records)} 条记录")
            else:
                # 查找district为空、NaN或缺失的记录
                missing_mask = (
                    self.data['district'].isna() | 
                    (self.data['district'] == '') | 
                    (self.data['district'] == 'nan') |
                    (self.data['district'].astype(str).str.strip() == '')
                )
                
                self.missing_records = self.data[missing_mask].index.tolist()
                
                self.logger.info(f"   - 找到 {len(self.missing_records)} 条缺失district信息的记录")
                self.logger.info(f"   - 完整记录数: {len(self.data) - len(self.missing_records)}")
            
            if len(self.missing_records) == 0:
                self.logger.info("🎉 所有记录的district信息都已完整!")
                return False
            
            # 显示部分缺失记录的示例
            if len(self.missing_records) > 0:
                sample_indices = self.missing_records[:5]
                self.logger.info("   缺失记录示例:")
                for idx in sample_indices:
                    row = self.data.loc[idx]
                    station_name = row.get('station_name', f'Index_{idx}')
                    coords = f"({row['longitude']}, {row['latitude']})"
                    district_value = row.get('district', 'N/A')
                    self.logger.info(f"     - {station_name} {coords} district='{district_value}'")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 查找缺失记录失败: {e}")
            return False
    
    def load_progress(self):
        """加载进度文件"""
        try:
            if Path(self.progress_csv).exists():
                progress_df = pd.read_csv(self.progress_csv)
                self.processed_records = set(progress_df['index'].tolist())
                
                # 加载已更新的数据
                for _, row in progress_df.iterrows():
                    idx = row['index']
                    if row['status'] == 'success':
                        self.updated_data[idx] = {
                            'district': row.get('district', ''),
                            'city': row.get('city', ''),
                            'province': row.get('province', ''),
                            'street': row.get('street', ''),
                            'formatted_address': row.get('formatted_address', ''),
                            'api_status': 'success'
                        }
                
                self.logger.info(f"📈 加载进度文件: 已处理 {len(self.processed_records)} 条记录")
            else:
                self.logger.info("📈 未找到进度文件，从头开始处理")
        except Exception as e:
            self.logger.warning(f"⚠️ 加载进度文件失败: {e}")
            self.processed_records = set()
    
    def save_progress(self, index, district_info):
        """保存进度（线程安全）"""
        with self.write_lock:
            try:
                # 准备进度记录
                progress_record = {
                    'index': index,
                    'timestamp': datetime.now().isoformat(),
                    'district': district_info.get('district', ''),
                    'city': district_info.get('city', ''),
                    'province': district_info.get('province', ''),
                    'street': district_info.get('street', ''),
                    'formatted_address': district_info.get('formatted_address', ''),
                    'status': district_info.get('status', 'success')
                }
                
                # 写入进度文件
                progress_df = pd.DataFrame([progress_record])
                
                if Path(self.progress_csv).exists():
                    progress_df.to_csv(self.progress_csv, mode='a', header=False, index=False, encoding='utf-8-sig')
                else:
                    progress_df.to_csv(self.progress_csv, index=False, encoding='utf-8-sig')
                
                self.processed_records.add(index)
                
                # 如果成功，保存更新数据
                if district_info.get('status') == 'success':
                    self.updated_data[index] = {
                        'district': district_info.get('district', ''),
                        'city': district_info.get('city', ''),
                        'province': district_info.get('province', ''),
                        'street': district_info.get('street', ''),
                        'formatted_address': district_info.get('formatted_address', ''),
                        'api_status': 'success'
                    }
                
            except Exception as e:
                self.logger.error(f"❌ 保存进度失败: {e}")
    
    def rate_limit(self):
        """限制请求频率（线程安全）"""
        with self.request_lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            
            if time_since_last < REQUEST_INTERVAL:
                sleep_time = REQUEST_INTERVAL - time_since_last
                time.sleep(sleep_time)
            
            self.last_request_time = time.time()
    
    def call_baidu_api(self, longitude, latitude):
        """调用百度API"""
        self.rate_limit()
        
        try:
            # 构建请求参数
            params = {
                'ak': self.baidu_ak,
                'location': f"{latitude},{longitude}",  # 注意：百度API要求纬度在前
                **API_PARAMS
            }
            
            # 发送请求
            response = requests.get(BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            
            # 解析响应
            result = response.json()
            
            if result.get('status') == 0:  # 成功
                address_component = result.get('result', {}).get('addressComponent', {})
                
                district_info = {
                    'status': 'success',
                    'district': address_component.get('district', ''),
                    'city': address_component.get('city', ''),
                    'province': address_component.get('province', ''),
                    'street': address_component.get('street', ''),
                    'formatted_address': result.get('result', {}).get('formatted_address', ''),
                    'api_response': json.dumps(result, ensure_ascii=False)
                }
                
                self.logger.info(f"✅ API调用成功: {longitude}, {latitude} -> {district_info['district']}")
                return district_info
                
            else:
                error_msg = result.get('message', 'Unknown error')
                self.logger.warning(f"⚠️ API返回错误: {error_msg}")
                return {
                    'status': 'api_error',
                    'error_message': error_msg,
                    'district': '',
                    'city': '',
                    'province': '',
                    'street': '',
                    'formatted_address': ''
                }
                
        except requests.RequestException as e:
            self.logger.error(f"❌ 网络请求失败: {e}")
            return {
                'status': 'network_error',
                'error_message': str(e),
                'district': '',
                'city': '',
                'province': '',
                'street': '',
                'formatted_address': ''
            }
        except Exception as e:
            self.logger.error(f"❌ API调用异常: {e}")
            return {
                'status': 'exception',
                'error_message': str(e),
                'district': '',
                'city': '',
                'province': '',
                'street': '',
                'formatted_address': ''
            }
    
    def process_single_record(self, index):
        """处理单条记录"""
        try:
            # 检查是否已处理
            if index in self.processed_records:
                self.logger.info(f"⏭️ 跳过已处理记录: {index}")
                return True
            
            row = self.data.loc[index]
            longitude = row['longitude']
            latitude = row['latitude']
            station_name = row.get('station_name', f'Index_{index}')
            
            self.logger.info(f"🔄 正在处理记录 {index}: {station_name} ({longitude}, {latitude})")
            
            # 调用API
            district_info = self.call_baidu_api(longitude, latitude)
            
            # 保存进度
            self.save_progress(index, district_info)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 处理记录 {index} 失败: {e}")
            return False
    
    def run_concurrent_processing(self):
        """运行并发处理"""
        self.logger.info("🚀 开始并发处理缺失的记录...")
        
        # 加载进度
        self.load_progress()
        
        # 过滤未处理的记录
        unprocessed_indices = [idx for idx in self.missing_records if idx not in self.processed_records]
        
        if not unprocessed_indices:
            self.logger.info("🎉 所有缺失记录已处理完成！")
            return True
        
        self.logger.info(f"📊 待处理记录数: {len(unprocessed_indices)}")
        
        # 并发处理
        success_count = 0
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # 提交任务
            future_to_index = {}
            for index in unprocessed_indices:
                future = executor.submit(self.process_single_record, index)
                future_to_index[future] = index
            
            # 处理结果
            completed_count = 0
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                
                try:
                    success = future.result()
                    if success:
                        success_count += 1
                    
                    completed_count += 1
                    
                    # 显示进度
                    progress = completed_count / len(unprocessed_indices) * 100
                    self.logger.info(f"📈 进度: {completed_count}/{len(unprocessed_indices)} ({progress:.1f}%) - 成功: {success_count}")
                    
                except Exception as e:
                    self.logger.error(f"❌ 处理任务 {index} 时发生异常: {e}")
        
        self.logger.info(f"🎉 并发处理完成！成功处理: {success_count}/{len(unprocessed_indices)}")
        return True
    
    def update_original_data(self):
        """更新原始数据"""
        try:
            self.logger.info("📝 正在更新原始数据...")
            
            # 创建数据副本
            updated_data = self.data.copy()
            
            # 确保必要的列存在
            required_columns = ['district', 'city', 'province', 'street', 'formatted_address', 'api_status']
            for col in required_columns:
                if col not in updated_data.columns:
                    updated_data[col] = ''
            
            # 更新数据
            update_count = 0
            for index, update_info in self.updated_data.items():
                if index in updated_data.index:
                    for key, value in update_info.items():
                        updated_data.loc[index, key] = value
                    update_count += 1
            
            # 保存更新后的数据
            updated_data.to_csv(self.output_csv, index=False, encoding='utf-8-sig')
            
            self.logger.info(f"✅ 数据更新完成!")
            self.logger.info(f"   - 更新记录数: {update_count}")
            self.logger.info(f"   - 输出文件: {self.output_csv}")
            
            # 统计更新后的完整性
            if 'district' in updated_data.columns:
                complete_count = len(updated_data[
                    ~(updated_data['district'].isna() | 
                      (updated_data['district'] == '') | 
                      (updated_data['district'] == 'nan'))
                ])
                total_count = len(updated_data)
                completion_rate = (complete_count / total_count) * 100
                
                self.logger.info(f"   - 数据完整性: {complete_count}/{total_count} ({completion_rate:.1f}%)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 更新数据失败: {e}")
            return False
    
    def run(self):
        """运行主程序"""
        try:
            self.logger.info("🚀 开始运行区域信息补充程序...")
            
            # 检查API Key
            if self.baidu_ak == "您的ak" or not self.baidu_ak:
                self.logger.error("❌ 请先设置正确的百度地图API Key")
                return False
            
            # 加载数据
            if not self.load_data():
                return False
            
            # 查找缺失记录
            if not self.find_missing_records():
                return True  # 没有缺失记录
            
            # 并发处理
            if not self.run_concurrent_processing():
                return False
            
            # 更新原始数据
            if not self.update_original_data():
                return False
            
            self.logger.info("🎉 程序运行完成！")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 程序运行失败: {e}")
            return False

def main():
    """主函数"""
    print("🚀 区域信息补充程序")
    print("用于补充final_station_ranking_with_district.csv中缺失的district信息")
    print("="*60)
    
    # 检查输入文件
    if not Path(INPUT_CSV).exists():
        print(f"❌ 错误: 输入文件不存在: {INPUT_CSV}")
        return
    
    # 创建处理器
    completer = DistrictCompleter(
        baidu_ak=BAIDU_AK,
        input_csv=INPUT_CSV,
        output_csv=OUTPUT_CSV,
        progress_csv=PROGRESS_CSV
    )
    
    # 运行程序
    success = completer.run()
    
    if success:
        print("\n" + "="*60)
        print("🎉 程序执行完成！")
        print("="*60)
        print(f"输出文件:")
        print(f"  1. {OUTPUT_CSV} - 补充完整的数据文件")
        print(f"  2. {PROGRESS_CSV} - 处理进度文件")
        print(f"  3. district_complete.log - 日志文件")
        print(f"\n💡 使用说明:")
        print(f"  - 支持断点继续：重新运行程序会自动跳过已处理的记录")
        print(f"  - 高并发处理：程序使用{MAX_WORKERS}个并发线程加速处理")
        print(f"  - 只处理缺失数据：完整的记录不会重复处理")
    else:
        print("\n❌ 程序执行失败，请检查日志文件 district_complete.log")

if __name__ == "__main__":
    main() 