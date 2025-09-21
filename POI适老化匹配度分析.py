#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
医院与适老化设施匹配度分析
专门分析医院POI与公交适老化站点的空间匹配关系
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置区域 ====================
# 输入文件配置
BUS_STATIONS_CSV = "/Users/songdingan/Downloads/detect_bus/final_station_ranking_complete.csv"  # 公交站点数据
HOSPITAL_EXCEL = "/Users/songdingan/Downloads/detect_bus/医院.xlsx"  # 医院Excel数据
OUTPUT_DIR = "/Users/songdingan/Downloads/detect_bus/"

# 医院POI分析参数
HOSPITAL_CONFIG = {
    'search_radius': 800,  # 搜索半径(米) - 医院周边合理步行距离
    'weight': 1.0,         # 权重：医院是老年人最重要的出行目的地
    'poi_type': '医院'      # POI类型统一为医院
}

# 分析参数
HIGH_SCORE_PERCENTILE = 75  # 高分站点：前25%
LOW_SCORE_PERCENTILE = 25   # 低分站点：后25%
DISTANCE_THRESHOLD = 500    # 默认距离阈值(米)
GRID_SIZE = 0.01           # 网格分析单元大小(度)
# ==================================================

class HospitalAdaptabilityMatcher:
    """医院与适老化设施匹配度分析器"""
    
    def __init__(self, bus_stations_csv, hospital_excel, output_dir):
        self.bus_stations_csv = bus_stations_csv
        self.hospital_excel = hospital_excel
        self.output_dir = Path(output_dir)
        self.bus_data = None
        self.poi_data = None
        self.analysis_results = {}
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_data(self):
        """加载公交站点和医院数据"""
        try:
            print("📂 正在加载数据...")
            
            # 加载公交站点数据
            if Path(self.bus_stations_csv).exists():
                self.bus_data = pd.read_csv(self.bus_stations_csv)
                print(f"✅ 公交站点数据加载成功: {len(self.bus_data)}个站点")
            else:
                print(f"❌ 公交站点文件不存在: {self.bus_stations_csv}")
                return False
            
            # 加载医院Excel数据
            if Path(self.hospital_excel).exists():
                self.poi_data = pd.read_excel(self.hospital_excel)
                print(f"✅ 医院数据加载成功: {len(self.poi_data)}个医院POI")
            else:
                print(f"❌ 医院文件不存在: {self.hospital_excel}")
                return False
            
            # 数据预处理
            self._preprocess_data()
            
            return True
            
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            return False
    
    def _preprocess_data(self):
        """数据预处理"""
        try:
            print("\n🔧 正在进行数据预处理...")
            
            # 清理公交站点数据
            self.bus_data = self.bus_data[
                (self.bus_data['longitude'].notna()) & 
                (self.bus_data['latitude'].notna()) &
                (self.bus_data['comprehensive_score'].notna())
            ].copy()
            
            # 清理医院数据 - 只使用wgs84_lng和wgs84_lat字段
            if 'wgs84_lng' in self.poi_data.columns and 'wgs84_lat' in self.poi_data.columns:
                self.poi_data = self.poi_data[
                    (self.poi_data['wgs84_lng'].notna()) & 
                    (self.poi_data['wgs84_lat'].notna())
                ].copy()
                
                # 标准化列名为longitude和latitude
                self.poi_data['longitude'] = self.poi_data['wgs84_lng']
                self.poi_data['latitude'] = self.poi_data['wgs84_lat']
            else:
                print(f"❌ 医院数据缺少wgs84_lng或wgs84_lat字段")
                raise ValueError("医院数据格式不正确")
            
            # 分类POI
            self._classify_poi()
            
            # 计算适老化分级
            self._calculate_adaptability_levels()
            
            print(f"   - 有效公交站点: {len(self.bus_data)}个")
            print(f"   - 有效医院POI: {len(self.poi_data)}个")
            
        except Exception as e:
            print(f"❌ 数据预处理失败: {e}")
            raise
    
    def _classify_poi(self):
        """将所有POI统一作为医院处理"""
        try:
            print("   🏥 正在处理医院POI数据...")
            
            # 所有POI统一作为医院处理
            self.poi_data['elderly_poi_type'] = HOSPITAL_CONFIG['poi_type']
            self.poi_data['elderly_poi_weight'] = HOSPITAL_CONFIG['weight']
            self.poi_data['search_radius'] = HOSPITAL_CONFIG['search_radius']
            
            # 直接使用所有POI作为医院
            self.elderly_poi = self.poi_data.copy()
            
            print(f"   ✅ 医院POI总数: {len(self.elderly_poi)}个")
            print(f"   - 搜索半径: {HOSPITAL_CONFIG['search_radius']}米")
            
        except Exception as e:
            print(f"❌ POI处理失败: {e}")
            raise
    
    def _calculate_adaptability_levels(self):
        """计算适老化分级"""
        try:
            print("   📊 正在计算适老化分级...")
            
            # 计算分位数阈值
            high_threshold = self.bus_data['comprehensive_score'].quantile(HIGH_SCORE_PERCENTILE / 100)
            low_threshold = self.bus_data['comprehensive_score'].quantile(LOW_SCORE_PERCENTILE / 100)
            
            # 分级
            def classify_adaptability(score):
                if score >= high_threshold:
                    return 'High'
                elif score <= low_threshold:
                    return 'Low'
                else:
                    return 'Medium'
            
            self.bus_data['adaptability_level'] = self.bus_data['comprehensive_score'].apply(classify_adaptability)
            
            # 统计各级别数量
            level_counts = self.bus_data['adaptability_level'].value_counts()
            
            print(f"     - 高适老化站点 (前25%): {level_counts.get('High', 0)}个, 阈值≥{high_threshold:.4f}")
            print(f"     - 低适老化站点 (后25%): {level_counts.get('Low', 0)}个, 阈值≤{low_threshold:.4f}")
            print(f"     - 中等适老化站点: {level_counts.get('Medium', 0)}个")
            
            self.analysis_results['thresholds'] = {
                'high_threshold': high_threshold,
                'low_threshold': low_threshold
            }
            
        except Exception as e:
            print(f"❌ 适老化分级失败: {e}")
            raise
    
    def calculate_spatial_matching(self):
        """计算空间匹配度"""
        try:
            print("\n🗺️ 正在计算空间匹配度...")
            
            matching_results = []
            
            # 分析医院POI
            poi_type = HOSPITAL_CONFIG['poi_type'] 
            search_radius = HOSPITAL_CONFIG['search_radius']
            
            print(f"\n   分析{poi_type}POI...")
            poi_subset = self.elderly_poi  # 所有POI都是医院
            
            # 按适老化等级分析
            for level in ['High', 'Low', 'Medium']:
                bus_subset = self.bus_data[self.bus_data['adaptability_level'] == level]
                if len(bus_subset) == 0:
                    continue
                
                # 计算距离矩阵
                bus_coords = bus_subset[['longitude', 'latitude']].values
                poi_coords = poi_subset[['longitude', 'latitude']].values
                
                # 计算距离 (使用简化的欧几里得距离，转换为米)
                distances = self._calculate_distance_matrix(bus_coords, poi_coords)
                
                # 计算匹配指标
                near_poi_count = 0
                total_distance = 0
                matched_pairs = 0
                
                for i, bus_row in bus_subset.iterrows():
                    min_distances = distances[bus_subset.index.get_loc(i)]
                    nearby_pois = np.sum(min_distances <= search_radius)
                    
                    if nearby_pois > 0:
                        near_poi_count += 1
                        min_dist = np.min(min_distances)
                        total_distance += min_dist
                        matched_pairs += nearby_pois
                
                # 计算匹配度指标
                coverage_rate = near_poi_count / len(bus_subset) if len(bus_subset) > 0 else 0
                avg_distance = total_distance / near_poi_count if near_poi_count > 0 else np.inf
                poi_accessibility = matched_pairs / len(poi_subset) if len(poi_subset) > 0 else 0
                
                matching_results.append({
                    'POI_Type': poi_type,
                    'Adaptability_Level': level,
                    'Bus_Stations_Count': len(bus_subset),
                    'POI_Count': len(poi_subset),
                    'Stations_Near_POI': near_poi_count,
                    'Coverage_Rate': coverage_rate,
                    'Average_Distance': avg_distance,
                    'POI_Accessibility': poi_accessibility,
                    'Search_Radius': search_radius
                })
                
                print(f"     - {level}级站点: 覆盖率{coverage_rate:.2%}, 平均距离{avg_distance:.0f}m")
            
            self.analysis_results['spatial_matching'] = pd.DataFrame(matching_results)
            
            # 识别供需缺口
            self._identify_service_gaps()
            
            return True
            
        except Exception as e:
            print(f"❌ 空间匹配度计算失败: {e}")
            return False
    
    def _calculate_distance_matrix(self, coords1, coords2):
        """计算距离矩阵 (米)"""
        # 简化的距离计算：1度约等于111000米
        # 对于上海地区的小范围分析，这个近似是可接受的
        
        # 将经纬度差异转换为米
        coords1_m = coords1 * [111000 * np.cos(np.radians(31.2)), 111000]  # 上海纬度约31.2度
        coords2_m = coords2 * [111000 * np.cos(np.radians(31.2)), 111000]
        
        # 计算欧几里得距离
        distances = cdist(coords1_m, coords2_m, metric='euclidean')
        
        return distances
    
    def _identify_service_gaps(self):
        """识别服务缺口"""
        try:
            print("\n🔍 正在识别服务缺口...")
            
            matching_df = self.analysis_results['spatial_matching']
            
            service_gaps = []
            
            # 分析医院POI的服务缺口
            poi_type = HOSPITAL_CONFIG['poi_type']
            poi_data = matching_df[matching_df['POI_Type'] == poi_type]
            
            if len(poi_data) > 0:
                # 获取高低适老化等级的数据
                high_data = poi_data[poi_data['Adaptability_Level'] == 'High']
                low_data = poi_data[poi_data['Adaptability_Level'] == 'Low']
                
                if len(high_data) > 0 and len(low_data) > 0:
                    high_coverage = high_data['Coverage_Rate'].iloc[0]
                    low_coverage = low_data['Coverage_Rate'].iloc[0]
                    
                    # 计算缺口指标
                    coverage_gap = high_coverage - low_coverage
                    
                    high_distance = high_data['Average_Distance'].iloc[0]
                    low_distance = low_data['Average_Distance'].iloc[0]
                    
                    distance_gap = low_distance - high_distance if not np.isinf(low_distance) else np.inf
                    
                    gap_severity = 'High' if coverage_gap > 0.2 or distance_gap > 300 else \
                                  'Medium' if coverage_gap > 0.1 or distance_gap > 150 else 'Low'
                    
                    service_gaps.append({
                        'POI_Type': poi_type,
                        'High_Coverage_Rate': high_coverage,
                        'Low_Coverage_Rate': low_coverage,
                        'Coverage_Gap': coverage_gap,
                        'High_Avg_Distance': high_distance,
                        'Low_Avg_Distance': low_distance,
                        'Distance_Gap': distance_gap,
                        'Gap_Severity': gap_severity,
                        'Priority_Level': HOSPITAL_CONFIG['weight']
                    })
                    
                    print(f"   - {poi_type}: 覆盖率差距{coverage_gap:.2%}, 严重程度{gap_severity}")
            
            self.analysis_results['service_gaps'] = pd.DataFrame(service_gaps)
            
            # 按优先级排序
            if len(service_gaps) > 0:
                self.analysis_results['service_gaps'] = self.analysis_results['service_gaps'].sort_values(
                    ['Gap_Severity', 'Priority_Level'], 
                    ascending=[False, False]
                )
            
        except Exception as e:
            print(f"❌ 服务缺口识别失败: {e}")
    
    def calculate_poi_density_analysis(self):
        """计算POI密度与适老化覆盖强度分析"""
        try:
            print("\n📍 正在进行POI密度分析...")
            
            # 创建空间网格
            lon_min = min(self.bus_data['longitude'].min(), self.elderly_poi['longitude'].min())
            lon_max = max(self.bus_data['longitude'].max(), self.elderly_poi['longitude'].max())
            lat_min = min(self.bus_data['latitude'].min(), self.elderly_poi['latitude'].min())
            lat_max = max(self.bus_data['latitude'].max(), self.elderly_poi['latitude'].max())
            
            # 创建网格点
            lon_grid = np.arange(lon_min, lon_max + GRID_SIZE, GRID_SIZE)
            lat_grid = np.arange(lat_min, lat_max + GRID_SIZE, GRID_SIZE)
            
            density_results = []
            
            for i, lon in enumerate(lon_grid[:-1]):
                for j, lat in enumerate(lat_grid[:-1]):
                    # 定义网格范围
                    grid_bounds = {
                        'lon_min': lon,
                        'lon_max': lon + GRID_SIZE,
                        'lat_min': lat,
                        'lat_max': lat + GRID_SIZE
                    }
                    
                    # 计算网格内POI密度
                    poi_in_grid = self.elderly_poi[
                        (self.elderly_poi['longitude'] >= grid_bounds['lon_min']) &
                        (self.elderly_poi['longitude'] < grid_bounds['lon_max']) &
                        (self.elderly_poi['latitude'] >= grid_bounds['lat_min']) &
                        (self.elderly_poi['latitude'] < grid_bounds['lat_max'])
                    ]
                    
                    # 计算网格内适老化站点情况
                    bus_in_grid = self.bus_data[
                        (self.bus_data['longitude'] >= grid_bounds['lon_min']) &
                        (self.bus_data['longitude'] < grid_bounds['lon_max']) &
                        (self.bus_data['latitude'] >= grid_bounds['lat_min']) &
                        (self.bus_data['latitude'] < grid_bounds['lat_max'])
                    ]
                    
                    if len(poi_in_grid) > 0 or len(bus_in_grid) > 0:
                        # 计算密度指标
                        grid_area = GRID_SIZE * GRID_SIZE * (111000 ** 2)  # 转换为平方米
                        poi_density = len(poi_in_grid) / (grid_area / 1000000)  # 每平方公里POI数量
                        
                        # 计算适老化指标
                        high_stations = len(bus_in_grid[bus_in_grid['adaptability_level'] == 'High'])
                        total_stations = len(bus_in_grid)
                        adaptability_intensity = high_stations / total_stations if total_stations > 0 else 0
                        avg_score = bus_in_grid['comprehensive_score'].mean() if total_stations > 0 else 0
                        
                        # 按POI类型统计
                        poi_type_counts = poi_in_grid['elderly_poi_type'].value_counts().to_dict()
                        
                        density_results.append({
                            'Grid_ID': f"{i}_{j}",
                            'Center_Longitude': lon + GRID_SIZE/2,
                            'Center_Latitude': lat + GRID_SIZE/2,
                            'POI_Density': poi_density,
                            'Total_POI': len(poi_in_grid),
                            'Total_Stations': total_stations,
                            'High_Adaptability_Stations': high_stations,
                            'Adaptability_Intensity': adaptability_intensity,
                            'Average_Adaptability_Score': avg_score,
                            **poi_type_counts
                        })
            
            self.analysis_results['density_analysis'] = pd.DataFrame(density_results)
            
            # 计算相关性
            if len(density_results) > 2:
                density_df = self.analysis_results['density_analysis']
                correlation = density_df['POI_Density'].corr(density_df['Adaptability_Intensity'])
                
                print(f"   ✅ POI密度与适老化强度相关性: r = {correlation:.3f}")
                self.analysis_results['poi_adaptability_correlation'] = correlation
            
            return True
            
        except Exception as e:
            print(f"❌ POI密度分析失败: {e}")
            return False
    

    

    
    def save_results(self):
        """保存分析结果"""
        try:
            print("\n💾 正在保存分析结果...")
            
            # 保存空间匹配结果
            if 'spatial_matching' in self.analysis_results:
                matching_file = self.output_dir / "hospital_spatial_matching_results.csv"
                self.analysis_results['spatial_matching'].to_csv(matching_file, index=False, encoding='utf-8-sig')
                print(f"✅ 空间匹配结果已保存: {matching_file}")
            
            # 保存服务缺口分析
            if 'service_gaps' in self.analysis_results:
                gaps_file = self.output_dir / "hospital_service_gaps_analysis.csv"
                self.analysis_results['service_gaps'].to_csv(gaps_file, index=False, encoding='utf-8-sig')
                print(f"✅ 服务缺口分析已保存: {gaps_file}")
            
            # 保存医院密度分析
            if 'density_analysis' in self.analysis_results:
                density_file = self.output_dir / "hospital_density_analysis.csv"
                self.analysis_results['density_analysis'].to_csv(density_file, index=False, encoding='utf-8-sig')
                print(f"✅ 医院密度分析已保存: {density_file}")
            
            # 保存医院数据
            hospital_classified_file = self.output_dir / "hospital_data_processed.csv"
            self.elderly_poi.to_csv(hospital_classified_file, index=False, encoding='utf-8-sig')
            print(f"✅ 医院数据已保存: {hospital_classified_file}")
            
            return True
            
        except Exception as e:
            print(f"❌ 保存结果失败: {e}")
            return False
    
    def run_analysis(self):
        """运行完整分析"""
        try:
            print("🚀 开始医院与适老化设施匹配度分析...")
            print("="*60)
            
            # 1. 加载数据
            if not self.load_data():
                return False
            
            # 2. 计算空间匹配度
            if not self.calculate_spatial_matching():
                return False
            
            # 3. POI密度分析
            if not self.calculate_poi_density_analysis():
                return False
            
            # 4. 保存结果
            if not self.save_results():
                return False
            
            print("\n" + "="*60)
            print("🎉 医院与适老化设施匹配度分析完成！")
            
            # 显示核心结果
            self._display_summary()
            
            return True
            
        except Exception as e:
            print(f"❌ 分析失败: {e}")
            return False
    
    def _display_summary(self):
        """显示分析摘要"""
        try:
            print("\n📊 核心分析结果:")
            print("="*50)
            
            # 医院统计
            hospital_count = len(self.elderly_poi)
            print(f"🏥 医院POI总数: {hospital_count}个")
            
            # 相关性结果
            if 'poi_adaptability_correlation' in self.analysis_results:
                correlation = self.analysis_results['poi_adaptability_correlation']
                print(f"\n🔗 医院密度与适老化强度相关性: r = {correlation:.3f}")
            
            # 服务缺口
            if 'service_gaps' in self.analysis_results and len(self.analysis_results['service_gaps']) > 0:
                gaps_df = self.analysis_results['service_gaps']
                high_gaps = gaps_df[gaps_df['Gap_Severity'] == 'High']
                
                print(f"\n🚨 高优先级改造区域:")
                if len(high_gaps) > 0:
                    for _, row in high_gaps.iterrows():
                        print(f"   - {row['POI_Type']}: 覆盖率差距{row['Coverage_Gap']:.2%}")
                else:
                    print("   - 暂无高优先级缺口区域")
            
            print("="*50)
            
        except Exception as e:
            print(f"❌ 显示摘要失败: {e}")

def main():
    """主函数"""
    print("🚀 医院与适老化设施匹配度分析")
    print("专门分析医院POI与公交站点适老化设施的空间匹配关系")
    print("="*60)
    
    # 检查输入文件
    if not Path(BUS_STATIONS_CSV).exists():
        print(f"❌ 错误: 公交站点文件不存在: {BUS_STATIONS_CSV}")
        return
    
    if not Path(HOSPITAL_EXCEL).exists():
        print(f"❌ 错误: 医院数据文件不存在: {HOSPITAL_EXCEL}")
        print("请将医院数据保存为医院.xlsx文件，程序将使用wgs84_lng和wgs84_lat字段")
        print("注意: 文件中的所有POI点都将被作为医院处理")
        return
    
    # 创建分析器
    analyzer = HospitalAdaptabilityMatcher(BUS_STATIONS_CSV, HOSPITAL_EXCEL, OUTPUT_DIR)
    
    # 运行分析
    success = analyzer.run_analysis()
    
    if success:
        print(f"\n📁 输出文件:")
        print(f"  1. hospital_spatial_matching_results.csv - 医院空间匹配结果")
        print(f"  2. hospital_service_gaps_analysis.csv - 医院服务缺口分析")
        print(f"  3. hospital_density_analysis.csv - 医院密度分析")
        print(f"  4. hospital_data_processed.csv - 处理后医院数据")
        print(f"\n所有文件已保存到: {OUTPUT_DIR}")
    else:
        print("\n❌ 分析失败，请查看错误信息")

if __name__ == "__main__":
    main() 