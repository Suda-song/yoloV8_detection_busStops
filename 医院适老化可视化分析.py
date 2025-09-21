#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
医院适老化设施匹配度可视化分析
基于POI分析结果生成多种可视化图表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和图表样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# ==================== 配置区域 ====================
# 输入文件路径
DATA_DIR = "/Users/songdingan/Downloads/detect_bus/"
SPATIAL_MATCHING_FILE = DATA_DIR + "hospital_spatial_matching_results.csv"
SERVICE_GAPS_FILE = DATA_DIR + "hospital_service_gaps_analysis.csv"
DENSITY_ANALYSIS_FILE = DATA_DIR + "hospital_density_analysis.csv"

# 输出目录
OUTPUT_DIR = "/Users/songdingan/Downloads/detect_bus/visualizations/"
# ==================================================

class HospitalVisualizationAnalyzer:
    """医院适老化设施可视化分析器"""
    
    def __init__(self):
        self.output_dir = Path(OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载数据
        self.spatial_data = None
        self.gaps_data = None
        self.density_data = None
        
    def load_data(self):
        """加载分析结果数据"""
        try:
            print("📂 正在加载分析结果数据...")
            
            # 加载空间匹配数据
            if Path(SPATIAL_MATCHING_FILE).exists():
                self.spatial_data = pd.read_csv(SPATIAL_MATCHING_FILE)
                print(f"✅ 空间匹配数据加载成功: {len(self.spatial_data)}条记录")
            else:
                print(f"❌ 空间匹配文件不存在: {SPATIAL_MATCHING_FILE}")
                return False
            
            # 加载服务缺口数据
            if Path(SERVICE_GAPS_FILE).exists():
                self.gaps_data = pd.read_csv(SERVICE_GAPS_FILE)
                print(f"✅ 服务缺口数据加载成功: {len(self.gaps_data)}条记录")
            else:
                print(f"❌ 服务缺口文件不存在: {SERVICE_GAPS_FILE}")
                return False
            
            # 加载密度分析数据
            if Path(DENSITY_ANALYSIS_FILE).exists():
                self.density_data = pd.read_csv(DENSITY_ANALYSIS_FILE)
                print(f"✅ 密度分析数据加载成功: {len(self.density_data)}条记录")
            else:
                print(f"❌ 密度分析文件不存在: {DENSITY_ANALYSIS_FILE}")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            return False
    
    def create_coverage_comparison_chart(self):
        """创建适老化等级覆盖率对比图"""
        try:
            print("\n📊 正在创建覆盖率对比图...")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # 1. 覆盖率柱状图
            levels = self.spatial_data['Adaptability_Level']
            coverage_rates = self.spatial_data['Coverage_Rate'] * 100
            colors = ['#ff4444', '#ffa500', '#4CAF50']  # Red, Orange, Green
            
            bars = ax1.bar(levels, coverage_rates, color=colors, alpha=0.8, edgecolor='black')
            ax1.set_title('Hospital Coverage Rate by Adaptability Level', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Adaptability Level', fontsize=12)
            ax1.set_ylabel('Coverage Rate (%)', fontsize=12)
            ax1.set_ylim(0, 100)
            
            # 添加数值标签
            for bar, rate in zip(bars, coverage_rates):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                        f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            ax1.grid(True, alpha=0.3, axis='y')
            
            # 2. 平均距离对比图
            distances = self.spatial_data['Average_Distance']
            finite_distances = distances[np.isfinite(distances)]
            finite_levels = levels[np.isfinite(distances)]
            
            bars2 = ax2.bar(finite_levels, finite_distances, color=colors[:len(finite_distances)], 
                           alpha=0.8, edgecolor='black')
            ax2.set_title('Average Distance to Hospitals by Adaptability Level', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Adaptability Level', fontsize=12)
            ax2.set_ylabel('Average Distance (meters)', fontsize=12)
            
            # 添加数值标签
            for bar, dist in zip(bars2, finite_distances):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                        f'{dist:.0f}m', ha='center', va='bottom', fontweight='bold')
            
            ax2.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            
            # 保存图表
            chart_file = self.output_dir / "coverage_comparison_chart.png"
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            print(f"✅ 覆盖率对比图已保存: {chart_file}")
            
            plt.show()
            
        except Exception as e:
            print(f"❌ 创建覆盖率对比图失败: {e}")
    
    def create_service_gap_analysis_chart(self):
        """创建服务缺口分析图"""
        try:
            print("\n📊 正在创建服务缺口分析图...")
            
            if len(self.gaps_data) == 0:
                print("⚠️ 无服务缺口数据，跳过该图表")
                return
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Hospital Service Gap Analysis', fontsize=16, fontweight='bold')
            
            # 1. 覆盖率差距
            gap_data = self.gaps_data.iloc[0]
            coverage_data = {
                'High Adaptability': gap_data['High_Coverage_Rate'] * 100,
                'Low Adaptability': gap_data['Low_Coverage_Rate'] * 100
            }
            
            bars1 = ax1.bar(coverage_data.keys(), coverage_data.values(), 
                           color=['#4CAF50', '#ff4444'], alpha=0.8)
            ax1.set_title('Coverage Rate Comparison', fontweight='bold')
            ax1.set_ylabel('Coverage Rate (%)')
            ax1.set_ylim(0, 100)
            
            for bar, value in zip(bars1, coverage_data.values()):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                        f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            # 2. 距离差距
            if not (np.isinf(gap_data['High_Avg_Distance']) or np.isinf(gap_data['Low_Avg_Distance'])):
                distance_data = {
                    'High Adaptability': gap_data['High_Avg_Distance'],
                    'Low Adaptability': gap_data['Low_Avg_Distance']
                }
                
                bars2 = ax2.bar(distance_data.keys(), distance_data.values(), 
                               color=['#4CAF50', '#ff4444'], alpha=0.8)
                ax2.set_title('Average Distance Comparison', fontweight='bold')
                ax2.set_ylabel('Average Distance (meters)')
                
                for bar, value in zip(bars2, distance_data.values()):
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                            f'{value:.0f}m', ha='center', va='bottom', fontweight='bold')
            else:
                ax2.text(0.5, 0.5, 'Distance data unavailable', ha='center', va='center', 
                        transform=ax2.transAxes, fontsize=12)
                ax2.set_title('Average Distance Comparison', fontweight='bold')
            
            # 3. 缺口严重程度
            severity = gap_data['Gap_Severity']
            severity_colors = {'High': '#ff4444', 'Medium': '#ffa500', 'Low': '#4CAF50'}
            
            ax3.pie([1], labels=[f'Gap Severity: {severity}'], 
                   colors=[severity_colors.get(severity, '#999999')],
                   autopct='', startangle=90)
            ax3.set_title('Service Gap Severity Level', fontweight='bold')
            
            # 4. 覆盖率差距数值
            coverage_gap = gap_data['Coverage_Gap'] * 100
            distance_gap = gap_data['Distance_Gap'] if not np.isinf(gap_data['Distance_Gap']) else 0
            
            metrics = ['Coverage Gap (%)', 'Distance Gap (m)']
            values = [coverage_gap, distance_gap]
            colors_bars = ['#2196F3', '#FF9800']
            
            bars4 = ax4.bar(metrics, values, color=colors_bars, alpha=0.8)
            ax4.set_title('Gap Metrics', fontweight='bold')
            ax4.set_ylabel('Gap Value')
            
            for bar, value in zip(bars4, values):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.02, 
                        f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            
            # 保存图表
            chart_file = self.output_dir / "service_gap_analysis_chart.png"
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            print(f"✅ 服务缺口分析图已保存: {chart_file}")
            
            plt.show()
            
        except Exception as e:
            print(f"❌ 创建服务缺口分析图失败: {e}")
    
    def create_density_heatmap(self):
        """创建医院密度与适老化强度热图"""
        try:
            print("\n📊 正在创建密度热图...")
            
            # 筛选有效数据
            valid_data = self.density_data[
                (self.density_data['Total_POI'] > 0) | 
                (self.density_data['Total_Stations'] > 0)
            ].copy()
            
            if len(valid_data) == 0:
                print("⚠️ 无有效密度数据，跳过热图")
                return
            
            # 创建子图
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Hospital Density and Adaptability Spatial Analysis', fontsize=16, fontweight='bold')
            
            # 1. 医院密度散点图
            scatter1 = ax1.scatter(valid_data['Center_Longitude'], valid_data['Center_Latitude'],
                                  c=valid_data['POI_Density'], s=valid_data['Total_POI']*3+10,
                                  cmap='Reds', alpha=0.6, edgecolors='black', linewidth=0.5)
            ax1.set_title('Hospital Density Distribution', fontweight='bold')
            ax1.set_xlabel('Longitude')
            ax1.set_ylabel('Latitude')
            plt.colorbar(scatter1, ax=ax1, label='POI Density (per km²)')
            
            # 2. 适老化强度散点图
            valid_adaptability = valid_data[valid_data['Total_Stations'] > 0]
            if len(valid_adaptability) > 0:
                scatter2 = ax2.scatter(valid_adaptability['Center_Longitude'], 
                                      valid_adaptability['Center_Latitude'],
                                      c=valid_adaptability['Adaptability_Intensity'], 
                                      s=valid_adaptability['Total_Stations']*5+10,
                                      cmap='Greens', alpha=0.6, edgecolors='black', linewidth=0.5)
                ax2.set_title('Adaptability Intensity Distribution', fontweight='bold')
                ax2.set_xlabel('Longitude')
                ax2.set_ylabel('Latitude')
                plt.colorbar(scatter2, ax=ax2, label='Adaptability Intensity')
            else:
                ax2.text(0.5, 0.5, 'No adaptability data available', ha='center', va='center', 
                        transform=ax2.transAxes, fontsize=12)
                ax2.set_title('Adaptability Intensity Distribution', fontweight='bold')
            
            # 3. 密度相关性分析
            correlation_data = valid_data[
                (valid_data['Total_Stations'] > 0) & 
                (valid_data['POI_Density'] > 0)
            ]
            
            if len(correlation_data) > 2:
                ax3.scatter(correlation_data['POI_Density'], 
                           correlation_data['Adaptability_Intensity'],
                           alpha=0.6, s=50, color='blue', edgecolors='black')
                
                # 添加趋势线
                if len(correlation_data) > 1:
                    z = np.polyfit(correlation_data['POI_Density'], 
                                  correlation_data['Adaptability_Intensity'], 1)
                    p = np.poly1d(z)
                    ax3.plot(correlation_data['POI_Density'], 
                            p(correlation_data['POI_Density']), "r--", alpha=0.8)
                
                # 计算相关系数
                corr = correlation_data['POI_Density'].corr(correlation_data['Adaptability_Intensity'])
                ax3.text(0.05, 0.95, f'Correlation: r={corr:.3f}', 
                        transform=ax3.transAxes, fontsize=10,
                        bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.7))
                
                ax3.set_title('POI Density vs Adaptability Intensity', fontweight='bold')
                ax3.set_xlabel('Hospital Density (per km²)')
                ax3.set_ylabel('Adaptability Intensity')
                ax3.grid(True, alpha=0.3)
            else:
                ax3.text(0.5, 0.5, 'Insufficient data for correlation analysis', 
                        ha='center', va='center', transform=ax3.transAxes, fontsize=12)
                ax3.set_title('POI Density vs Adaptability Intensity', fontweight='bold')
            
            # 4. 统计汇总
            stats_text = f"""
统计汇总 (Summary Statistics):

总网格数: {len(valid_data)}
有医院的网格: {len(valid_data[valid_data['Total_POI'] > 0])}
有公交站的网格: {len(valid_data[valid_data['Total_Stations'] > 0])}

平均医院密度: {valid_data['POI_Density'].mean():.2f} /km²
最大医院密度: {valid_data['POI_Density'].max():.2f} /km²

平均适老化强度: {valid_adaptability['Adaptability_Intensity'].mean():.3f if len(valid_adaptability) > 0 else 0}
            """
            
            ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round", facecolor='lightblue', alpha=0.7))
            ax4.set_title('Statistical Summary', fontweight='bold')
            ax4.axis('off')
            
            plt.tight_layout()
            
            # 保存图表
            chart_file = self.output_dir / "density_heatmap_analysis.png"
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            print(f"✅ 密度热图已保存: {chart_file}")
            
            plt.show()
            
        except Exception as e:
            print(f"❌ 创建密度热图失败: {e}")
    
    def create_comprehensive_dashboard(self):
        """创建综合仪表板"""
        try:
            print("\n📊 正在创建综合仪表板...")
            
            fig = plt.figure(figsize=(20, 12))
            fig.suptitle('Hospital-Bus Station Adaptability Comprehensive Dashboard', 
                        fontsize=18, fontweight='bold')
            
            # 创建网格布局
            gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
            
            # 1. 覆盖率环形图 (左上)
            ax1 = fig.add_subplot(gs[0, 0])
            if len(self.spatial_data) >= 3:
                coverage_rates = self.spatial_data['Coverage_Rate'] * 100
                labels = self.spatial_data['Adaptability_Level']
                colors = ['#ff4444', '#ffa500', '#4CAF50']
                
                wedges, texts, autotexts = ax1.pie(coverage_rates, labels=labels, 
                                                  autopct='%1.1f%%', startangle=90,
                                                  colors=colors, textprops={'fontsize': 9})
                ax1.set_title('Coverage Rate Distribution', fontweight='bold', fontsize=11)
            
            # 2. 站点数量对比 (中上)
            ax2 = fig.add_subplot(gs[0, 1])
            station_counts = self.spatial_data['Bus_Stations_Count']
            levels = self.spatial_data['Adaptability_Level']
            bars = ax2.bar(levels, station_counts, color=['#ff4444', '#ffa500', '#4CAF50'], alpha=0.8)
            ax2.set_title('Bus Stations by Adaptability Level', fontweight='bold', fontsize=11)
            ax2.set_ylabel('Number of Stations', fontsize=9)
            for bar, count in zip(bars, station_counts):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, 
                        str(count), ha='center', va='bottom', fontsize=9)
            
            # 3. 医院可达性 (右上)
            ax3 = fig.add_subplot(gs[0, 2])
            accessibility = self.spatial_data['POI_Accessibility']
            bars3 = ax3.bar(levels, accessibility, color=['#2196F3', '#9C27B0', '#FF5722'], alpha=0.8)
            ax3.set_title('Hospital Accessibility Index', fontweight='bold', fontsize=11)
            ax3.set_ylabel('Accessibility Index', fontsize=9)
            for bar, acc in zip(bars3, accessibility):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                        f'{acc:.2f}', ha='center', va='bottom', fontsize=9)
            
            # 4. 关键指标汇总 (右上角)
            ax4 = fig.add_subplot(gs[0, 3])
            
            # 计算关键指标
            total_hospitals = self.spatial_data['POI_Count'].iloc[0] if len(self.spatial_data) > 0 else 0
            total_stations = self.spatial_data['Bus_Stations_Count'].sum()
            avg_coverage = self.spatial_data['Coverage_Rate'].mean() * 100
            
            key_metrics = f"""
关键指标 (Key Metrics):

医院总数: {total_hospitals:,}
公交站总数: {total_stations:,}
平均覆盖率: {avg_coverage:.1f}%
搜索半径: 800m

分析日期: {pd.Timestamp.now().strftime('%Y-%m-%d')}
            """
            
            ax4.text(0.05, 0.95, key_metrics, transform=ax4.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round", facecolor='lightyellow', alpha=0.8))
            ax4.set_title('Key Performance Indicators', fontweight='bold', fontsize=11)
            ax4.axis('off')
            
            # 5. 密度分布热图 (下半部分左侧)
            ax5 = fig.add_subplot(gs[1:, :2])
            
            # 绘制密度热图
            valid_density = self.density_data[self.density_data['POI_Density'] > 0]
            if len(valid_density) > 0:
                scatter = ax5.scatter(valid_density['Center_Longitude'], 
                                     valid_density['Center_Latitude'],
                                     c=valid_density['POI_Density'], 
                                     s=valid_density['Total_POI']*2+5,
                                     cmap='YlOrRd', alpha=0.7, edgecolors='black', linewidth=0.3)
                ax5.set_title('Hospital Density Spatial Distribution', fontweight='bold', fontsize=12)
                ax5.set_xlabel('Longitude', fontsize=10)
                ax5.set_ylabel('Latitude', fontsize=10)
                
                # 添加颜色条
                cbar = plt.colorbar(scatter, ax=ax5, shrink=0.8)
                cbar.set_label('Hospital Density (per km²)', fontsize=9)
            
            # 6. 服务缺口分析 (下半部分右侧)
            ax6 = fig.add_subplot(gs[1, 2:])
            
            if len(self.gaps_data) > 0:
                gap_data = self.gaps_data.iloc[0]
                
                # 创建覆盖率对比
                coverage_comparison = {
                    'High\nAdaptability': gap_data['High_Coverage_Rate'] * 100,
                    'Low\nAdaptability': gap_data['Low_Coverage_Rate'] * 100
                }
                
                bars6 = ax6.bar(coverage_comparison.keys(), coverage_comparison.values(),
                               color=['#4CAF50', '#ff4444'], alpha=0.8, edgecolor='black')
                ax6.set_title('Service Coverage Gap Analysis', fontweight='bold', fontsize=12)
                ax6.set_ylabel('Coverage Rate (%)', fontsize=10)
                
                for bar, value in zip(bars6, coverage_comparison.values()):
                    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                            f'{value:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
                
                # 添加缺口信息
                gap_pct = gap_data['Coverage_Gap'] * 100
                severity = gap_data['Gap_Severity']
                ax6.text(0.5, 0.15, f'Gap: {gap_pct:.1f}%\nSeverity: {severity}', 
                        transform=ax6.transAxes, ha='center', fontsize=9,
                        bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.7))
            
            # 7. 统计趋势 (右下)
            ax7 = fig.add_subplot(gs[2, 2:])
            
            # 按适老化等级绘制多指标雷达图的替代版本
            metrics = ['Coverage Rate', 'Avg Distance', 'Accessibility']
            
            # 标准化数据用于对比
            coverage_norm = self.spatial_data['Coverage_Rate'] * 100
            distance_norm = 100 - (self.spatial_data['Average_Distance'] / 
                                  self.spatial_data['Average_Distance'].max() * 100)  # 距离越小越好
            access_norm = (self.spatial_data['POI_Accessibility'] / 
                          self.spatial_data['POI_Accessibility'].max() * 100)
            
            levels = self.spatial_data['Adaptability_Level']
            x_pos = np.arange(len(levels))
            
            width = 0.25
            bars1 = ax7.bar(x_pos - width, coverage_norm, width, label='Coverage Rate (%)', 
                           color='skyblue', alpha=0.8)
            bars2 = ax7.bar(x_pos, distance_norm, width, label='Distance Score', 
                           color='lightgreen', alpha=0.8)
            bars3 = ax7.bar(x_pos + width, access_norm, width, label='Accessibility Score', 
                           color='lightcoral', alpha=0.8)
            
            ax7.set_title('Multi-Metric Performance Comparison', fontweight='bold', fontsize=12)
            ax7.set_xlabel('Adaptability Level', fontsize=10)
            ax7.set_ylabel('Score (Normalized)', fontsize=10)
            ax7.set_xticks(x_pos)
            ax7.set_xticklabels(levels, fontsize=9)
            ax7.legend(fontsize=8)
            ax7.grid(True, alpha=0.3, axis='y')
            
            # 保存综合仪表板
            chart_file = self.output_dir / "comprehensive_dashboard.png"
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            print(f"✅ 综合仪表板已保存: {chart_file}")
            
            plt.show()
            
        except Exception as e:
            print(f"❌ 创建综合仪表板失败: {e}")
    
    def generate_summary_report(self):
        """生成可视化分析摘要报告"""
        try:
            print("\n📝 正在生成可视化分析摘要报告...")
            
            # 计算关键统计数据
            total_hospitals = self.spatial_data['POI_Count'].iloc[0] if len(self.spatial_data) > 0 else 0
            total_stations = self.spatial_data['Bus_Stations_Count'].sum()
            avg_coverage = self.spatial_data['Coverage_Rate'].mean() * 100
            
            # 密度分析统计
            valid_grids = len(self.density_data[self.density_data['POI_Density'] > 0])
            max_density = self.density_data['POI_Density'].max()
            
            # 生成报告
            report = f"""
# 医院适老化设施匹配度可视化分析报告

## 数据概览
- 分析日期: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
- 医院总数: {total_hospitals:,} 个
- 公交站总数: {total_stations:,} 个
- 平均覆盖率: {avg_coverage:.1f}%
- 搜索半径: 800米

## 适老化等级分析
"""
            
            for _, row in self.spatial_data.iterrows():
                level = row['Adaptability_Level']
                coverage = row['Coverage_Rate'] * 100
                stations = row['Bus_Stations_Count']
                avg_dist = row['Average_Distance']
                accessibility = row['POI_Accessibility']
                
                dist_text = f"{avg_dist:.0f}米" if not np.isinf(avg_dist) else "N/A"
                
                report += f"""
### {level}级适老化站点
- 站点数量: {stations} 个
- 医院覆盖率: {coverage:.1f}%
- 平均距离: {dist_text}
- 可达性指数: {accessibility:.2f}
"""
            
            # 服务缺口分析
            if len(self.gaps_data) > 0:
                gap_data = self.gaps_data.iloc[0]
                report += f"""
## 服务缺口分析
- 高低适老化覆盖率差距: {gap_data['Coverage_Gap']*100:.1f}%
- 缺口严重程度: {gap_data['Gap_Severity']}
- 优先级等级: {gap_data['Priority_Level']}
"""
            
            # 空间密度分析
            report += f"""
## 空间密度分析
- 有效分析网格: {valid_grids} 个
- 最大医院密度: {max_density:.1f} 个/平方公里
- 网格总数: {len(self.density_data)} 个
"""
            
            # 可视化文件列表
            report += f"""
## 生成的可视化文件
1. coverage_comparison_chart.png - 覆盖率对比图
2. service_gap_analysis_chart.png - 服务缺口分析图  
3. density_heatmap_analysis.png - 密度热图分析
4. comprehensive_dashboard.png - 综合仪表板
5. visualization_summary_report.md - 本分析报告

所有文件保存在: {self.output_dir}
"""
            
            # 保存报告
            report_file = self.output_dir / "visualization_summary_report.md"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)
            
            print(f"✅ 可视化分析摘要报告已保存: {report_file}")
            
        except Exception as e:
            print(f"❌ 生成摘要报告失败: {e}")
    
    def run_all_visualizations(self):
        """运行所有可视化分析"""
        try:
            print("🚀 开始医院适老化设施可视化分析...")
            print("="*60)
            
            # 1. 加载数据
            if not self.load_data():
                return False
            
            # 2. 创建各种图表
            self.create_coverage_comparison_chart()
            self.create_service_gap_analysis_chart()
            self.create_density_heatmap()
            self.create_comprehensive_dashboard()
            
            # 3. 生成摘要报告
            self.generate_summary_report()
            
            print("\n" + "="*60)
            print("🎉 医院适老化设施可视化分析完成！")
            print(f"\n📁 所有可视化文件已保存到: {self.output_dir}")
            
            return True
            
        except Exception as e:
            print(f"❌ 可视化分析失败: {e}")
            return False

def main():
    """主函数"""
    print("🎨 医院适老化设施匹配度可视化分析")
    print("基于POI分析结果生成多种可视化图表")
    print("="*60)
    
    # 检查输入文件
    required_files = [SPATIAL_MATCHING_FILE, SERVICE_GAPS_FILE, DENSITY_ANALYSIS_FILE]
    
    for file_path in required_files:
        if not Path(file_path).exists():
            print(f"❌ 错误: 必需文件不存在: {file_path}")
            print("请先运行POI适老化匹配度分析.py生成分析结果文件")
            return
    
    # 创建可视化分析器
    analyzer = HospitalVisualizationAnalyzer()
    
    # 运行所有可视化
    success = analyzer.run_all_visualizations()
    
    if success:
        print(f"\n📊 生成的可视化图表:")
        print(f"  1. coverage_comparison_chart.png - 适老化等级覆盖率对比")
        print(f"  2. service_gap_analysis_chart.png - 服务缺口详细分析")
        print(f"  3. density_heatmap_analysis.png - 医院密度与适老化强度空间分析")
        print(f"  4. comprehensive_dashboard.png - 综合仪表板")
        print(f"  5. visualization_summary_report.md - 可视化分析摘要报告")
    else:
        print("\n❌ 可视化分析失败，请查看错误信息")

if __name__ == "__main__":
    main() 