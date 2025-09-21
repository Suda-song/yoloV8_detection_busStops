#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通过百度逆地理编码API获取经纬度对应的行政区信息
支持并发请求、断点继续、线程安全写入
"""

import requests
import pandas as pd
import time
import threading
import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import logging
from datetime import datetime
import sys

# ==================== 配置区域 ====================
# 百度API配置
BAIDU_AK = "olQ5vPbNSFmY2xGL7mFZcIEbSQjbBCIh"  # 请替换为您的百度地图API Key
BASE_URL = "https://api.map.baidu.com/reverse_geocoding/v3/"

# 文件配置
INPUT_CSV = "/Users/songdingan/Downloads/detect_bus/final_station_ranking.csv"  # 输入文件
OUTPUT_CSV = "/Users/songdingan/Downloads/detect_bus/final_station_ranking_with_district.csv"  # 输出文件
PROGRESS_CSV = "/Users/songdingan/Downloads/detect_bus/api_progress.csv"  # 进度保存文件

# 测试配置
TEST_MODE = False  # 测试模式：True=只处理前几条记录，False=处理全部
TEST_RECORDS = 5  # 测试模式下处理的记录数

# 并发配置
MAX_WORKERS = 40  # 最大并发数（3 QPS）
REQUEST_INTERVAL = 1.0  # 请求间隔（秒），确保不超过3QPS

# API参数配置
API_PARAMS = {
    'extensions_poi': 1,
    'entire_poi': 1,
    'sort_strategy': 'distance',
    'output': 'json',
    'coordtype': 'bd09ll'
}
# ==================================================

class DistrictFetcher:
    """行政区信息获取器"""
    
    def __init__(self, baidu_ak, input_csv, output_csv, progress_csv):
        self.baidu_ak = baidu_ak
        self.input_csv = input_csv
        
        # 测试模式下使用不同的输出文件名
        if TEST_MODE:
            base_path = Path(output_csv).parent
            self.output_csv = base_path / f"test_{TEST_RECORDS}_records_with_district.csv"
            self.progress_csv = base_path / f"test_{TEST_RECORDS}_api_progress.csv"
        else:
            self.output_csv = output_csv
            self.progress_csv = progress_csv
        
        # 线程安全控制
        self.write_lock = threading.Lock()
        self.request_lock = threading.Lock()
        self.last_request_time = 0
        
        # 加载数据
        self.data = None
        self.processed_records = set()
        
        # 设置日志
        self.setup_logging()
        
    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('district_fetch.log', encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_data(self):
        """加载输入数据"""
        try:
            self.logger.info(f"📂 正在加载数据: {self.input_csv}")
            
            # 尝试不同的可能文件路径
            possible_paths = [
                self.input_csv,
                f"./{self.input_csv}",
                f"./train/{self.input_csv}",
                f"/Users/songdingan/Downloads/detect_bus/{self.input_csv}"
            ]
            
            for path in possible_paths:
                if Path(path).exists():
                    self.data = pd.read_csv(path)
                    self.input_csv = path
                    break
            else:
                raise FileNotFoundError(f"无法找到输入文件: {self.input_csv}")
            
            # 验证必要的列
            required_cols = ['longitude', 'latitude']
            missing_cols = [col for col in required_cols if col not in self.data.columns]
            if missing_cols:
                raise ValueError(f"输入文件缺少必要的列: {missing_cols}")
            
            # 测试模式：只处理前几条记录
            if TEST_MODE:
                original_len = len(self.data)
                self.data = self.data.head(TEST_RECORDS)
                self.logger.info(f"🧪 测试模式已启用: 只处理前 {TEST_RECORDS} 条记录 (原始数据: {original_len} 条)")
            
            self.logger.info(f"✅ 数据加载成功!")
            self.logger.info(f"   - 样本数量: {len(self.data)}")
            self.logger.info(f"   - 列名: {list(self.data.columns)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 数据加载失败: {e}")
            return False
    
    def load_progress(self):
        """加载进度文件"""
        try:
            if Path(self.progress_csv).exists():
                progress_df = pd.read_csv(self.progress_csv)
                self.processed_records = set(progress_df['index'].tolist())
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
                    'status': district_info.get('status', 'success')
                }
                
                # 写入进度文件
                progress_df = pd.DataFrame([progress_record])
                
                if Path(self.progress_csv).exists():
                    progress_df.to_csv(self.progress_csv, mode='a', header=False, index=False, encoding='utf-8-sig')
                else:
                    progress_df.to_csv(self.progress_csv, index=False, encoding='utf-8-sig')
                
                self.processed_records.add(index)
                
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
    
    def process_single_record(self, index, row):
        """处理单条记录"""
        try:
            # 检查是否已处理
            if index in self.processed_records:
                self.logger.info(f"⏭️ 跳过已处理记录: {index}")
                return None
            
            longitude = row['longitude']
            latitude = row['latitude']
            
            self.logger.info(f"🔄 正在处理记录 {index}: ({longitude}, {latitude})")
            
            # 调用API
            district_info = self.call_baidu_api(longitude, latitude)
            
            # 保存进度
            self.save_progress(index, district_info)
            
            # 返回结果用于最终写入
            result_row = row.copy()
            result_row['district'] = district_info['district']
            result_row['city'] = district_info['city']
            result_row['province'] = district_info['province']
            result_row['street'] = district_info['street']
            result_row['formatted_address'] = district_info['formatted_address']
            result_row['api_status'] = district_info['status']
            if 'error_message' in district_info:
                result_row['error_message'] = district_info['error_message']
            
            return result_row
            
        except Exception as e:
            self.logger.error(f"❌ 处理记录 {index} 失败: {e}")
            return None
    
    def write_final_result(self, processed_rows):
        """写入最终结果文件（线程安全）"""
        with self.write_lock:
            try:
                if not processed_rows:
                    return
                
                # 创建DataFrame
                result_df = pd.DataFrame(processed_rows)
                
                # 写入文件
                if Path(self.output_csv).exists():
                    # 追加模式
                    result_df.to_csv(self.output_csv, mode='a', header=False, index=False, encoding='utf-8-sig')
                else:
                    # 新建文件
                    result_df.to_csv(self.output_csv, index=False, encoding='utf-8-sig')
                
                self.logger.info(f"✅ 写入 {len(processed_rows)} 条结果到 {self.output_csv}")
                
            except Exception as e:
                self.logger.error(f"❌ 写入结果文件失败: {e}")
    
    def run_concurrent_processing(self):
        """运行并发处理"""
        self.logger.info("🚀 开始并发处理...")
        
        # 加载进度
        self.load_progress()
        
        # 过滤未处理的记录
        unprocessed_indices = []
        for index in self.data.index:
            if index not in self.processed_records:
                unprocessed_indices.append(index)
        
        if not unprocessed_indices:
            self.logger.info("🎉 所有记录已处理完成！")
            return True
        
        self.logger.info(f"📊 待处理记录数: {len(unprocessed_indices)}")
        
        # 并发处理
        processed_rows = []
        batch_size = 10  # 每批处理10条记录后写入一次
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # 提交任务
            future_to_index = {}
            for index in unprocessed_indices:
                row = self.data.loc[index]
                future = executor.submit(self.process_single_record, index, row)
                future_to_index[future] = index
            
            # 处理结果
            completed_count = 0
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                
                try:
                    result_row = future.result()
                    if result_row is not None:
                        processed_rows.append(result_row)
                    
                    completed_count += 1
                    
                    # 每处理完一批或全部完成时写入结果
                    if len(processed_rows) >= batch_size or completed_count == len(unprocessed_indices):
                        self.write_final_result(processed_rows)
                        processed_rows = []  # 清空已写入的数据
                    
                    # 显示进度
                    progress = completed_count / len(unprocessed_indices) * 100
                    self.logger.info(f"📈 进度: {completed_count}/{len(unprocessed_indices)} ({progress:.1f}%)")
                    
                except Exception as e:
                    self.logger.error(f"❌ 处理任务 {index} 时发生异常: {e}")
        
        self.logger.info("🎉 并发处理完成！")
        return True
    
    def create_final_merged_file(self):
        """创建最终合并文件"""
        try:
            self.logger.info("📝 创建最终合并文件...")
            
            # 如果输出文件已存在且包含所有记录，直接返回
            if Path(self.output_csv).exists():
                output_df = pd.read_csv(self.output_csv)
                if len(output_df) >= len(self.data):
                    self.logger.info(f"✅ 最终文件已存在且完整: {self.output_csv}")
                    return True
            
            # 读取进度文件
            if not Path(self.progress_csv).exists():
                self.logger.warning("⚠️ 进度文件不存在，无法创建最终文件")
                return False
            
            progress_df = pd.read_csv(self.progress_csv)
            
            # 合并原始数据和API结果
            merged_data = self.data.copy()
            
            # 添加新列
            merged_data['district'] = ''
            merged_data['city'] = ''
            merged_data['province'] = ''
            merged_data['street'] = ''
            merged_data['formatted_address'] = ''
            merged_data['api_status'] = ''
            
            # 填充API结果
            for _, progress_row in progress_df.iterrows():
                index = progress_row['index']
                if index in merged_data.index:
                    merged_data.loc[index, 'district'] = progress_row.get('district', '')
                    merged_data.loc[index, 'city'] = progress_row.get('city', '')
                    merged_data.loc[index, 'province'] = progress_row.get('province', '')
                    merged_data.loc[index, 'api_status'] = progress_row.get('status', '')
            
            # 保存最终文件
            merged_data.to_csv(self.output_csv, index=False, encoding='utf-8-sig')
            
            self.logger.info(f"✅ 最终文件创建成功: {self.output_csv}")
            self.logger.info(f"   - 总记录数: {len(merged_data)}")
            self.logger.info(f"   - 已获取区域信息的记录数: {len(merged_data[merged_data['district'] != ''])}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 创建最终文件失败: {e}")
            return False
    
    def run(self):
        """运行主程序"""
        try:
            self.logger.info("🚀 开始运行区域信息获取程序...")
            
            # 检查API Key
            if self.baidu_ak == "您的ak":
                self.logger.error("❌ 请先设置正确的百度地图API Key")
                return False
            
            # 加载数据
            if not self.load_data():
                return False
            
            # 并发处理
            if not self.run_concurrent_processing():
                return False
            
            # 创建最终文件
            if not self.create_final_merged_file():
                return False
            
            self.logger.info("🎉 程序运行完成！")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 程序运行失败: {e}")
            return False

def main():
    """主函数"""
    print("🚀 区域信息获取程序")
    if TEST_MODE:
        print(f"🧪 测试模式：只处理前 {TEST_RECORDS} 条记录")
    print("="*60)
    
    # 检查输入文件
    if not any(Path(p).exists() for p in [
        INPUT_CSV,
        f"./{INPUT_CSV}",
        f"./train/{INPUT_CSV}",
        f"/Users/songdingan/Downloads/detect_bus/{INPUT_CSV}"
    ]):
        print(f"❌ 错误: 输入文件不存在: {INPUT_CSV}")
        print("请确保final_station_ranking.csv文件存在")
        print("可能的位置:")
        print("- 当前目录")
        print("- ./train/目录")
        print("- /Users/songdingan/Downloads/detect_bus/目录")
        return
    
    # 创建处理器
    fetcher = DistrictFetcher(
        baidu_ak=BAIDU_AK,
        input_csv=INPUT_CSV,
        output_csv=OUTPUT_CSV,
        progress_csv=PROGRESS_CSV
    )
    
    # 运行程序
    success = fetcher.run()
    
    if success:
        print("\n" + "="*60)
        print("🎉 程序执行完成！")
        print("="*60)
        print(f"输出文件:")
        print(f"  1. {OUTPUT_CSV} - 包含区域信息的最终文件")
        print(f"  2. {PROGRESS_CSV} - 处理进度文件")
        print(f"  3. district_fetch.log - 日志文件")
        print(f"\n💡 使用说明:")
        print(f"  - 支持断点继续：重新运行程序会自动跳过已处理的记录")
        print(f"  - 并发安全：程序会自动控制请求频率和文件写入")
        print(f"  - 请确保设置正确的百度地图API Key")
    else:
        print("\n❌ 程序执行失败，请检查日志文件 district_fetch.log")

if __name__ == "__main__":
    main()
