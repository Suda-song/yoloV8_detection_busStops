#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
区域站点密度分析程序
计算各区每万人/平方公里的高分站点密度，结合人口老龄化率和空间密度
使用Z-score标准化消除量纲差异后进行跨区比较
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 配置区域 ====================
# 输入文件配置
INPUT_CSV = "/Users/songdingan/Downloads/detect_bus/final_station_ranking_complete.csv"  # 包含区域信息的站点数据
OUTPUT_DIR = "/Users/songdingan/Downloads/detect_bus/"  # 输出目录

# 高分站点定义
HIGH_SCORE_PERCENTILE = 75  # 综合得分分位数，排名前25%的站点认为是高分站点
HIGH_SCORE_THRESHOLD = None  # 动态计算的阈值，将在运行时确定

# 各区基础数据
DISTRICT_DATA = {
    '黄浦区': {'面积': 20.52, '人口数': 662030, '人口密度': 32262.67},
    '徐汇区': {'面积': 54.93, '人口数': 1113078, '人口密度': 20263.57},
    '长宁区': {'面积': 38.3, '人口数': 693051, '人口密度': 18095.33},
    '静安区': {'面积': 37.37, '人口数': 975707, '人口密度': 26109.37},
    '普陀区': {'面积': 55.53, '人口数': 1239800, '人口密度': 22326.67},
    '虹口区': {'面积': 23.45, '人口数': 757498, '人口密度': 32302.69},
    '杨浦区': {'面积': 60.61, '人口数': 1242548, '人口密度': 20500.71},
    '闵行区': {'面积': 372.56, '人口数': 2614218, '人口密度': 7016.90},
    '宝山区': {'面积': 271.3, '人口数': 1900809, '人口密度': 7006.3},
    '嘉定区': {'面积': 463.55, '人口数': 1161435, '人口密度': 2505.52},
    '浦东新区': {'面积': 1210.41, '人口数': 4619851, '人口密度': 3816.77},
    '金山区': {'面积': 613.28, '人口数': 279796, '人口密度': 456.23},
    '松江区': {'面积': 604.67, '人口数': 1304271, '人口密度': 2157.00},
    '青浦区': {'面积': 676.26, '人口数': 813128, '人口密度': 1202.39},
    '奉贤区': {'面积': 687.39, '人口数': 376669, '人口密度': 547.97}
}

# 人口老龄化率数据（2020年人口普查数据，65岁及以上人口比例%）
AGING_RATE = {
    '黄浦区': 23.8, '徐汇区': 22.1, '长宁区': 24.5, '静安区': 25.2, '普陀区': 20.8,
    '虹口区': 26.3, '杨浦区': 22.9, '闵行区': 15.6, '宝山区': 18.9, '嘉定区': 16.8,
    '浦东新区': 16.2, '金山区': 21.4, '松江区': 17.5, '青浦区': 18.3, '奉贤区': 19.2
}
# ==================================================

class DistrictDensityAnalyzer:
    """区域站点密度分析器"""
    
    def __init__(self, input_csv, output_dir):
        self.input_csv = input_csv
        self.output_dir = Path(output_dir)
        self.data = None
        self.analysis_results = {}
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_data(self):
        """加载站点数据"""
        try:
            print("📂 正在加载站点数据...")
            
            if not Path(self.input_csv).exists():
                raise FileNotFoundError(f"输入文件不存在: {self.input_csv}")
            
            self.data = pd.read_csv(self.input_csv)
            
            # 验证必要的列
            required_cols = ['district', 'comprehensive_score']
            missing_cols = [col for col in required_cols if col not in self.data.columns]
            if missing_cols:
                raise ValueError(f"输入文件缺少必要的列: {missing_cols}")
            
            print(f"✅ 数据加载成功!")
            print(f"   - 总站点数: {len(self.data)}")
            print(f"   - 列名: {list(self.data.columns)}")
            
            # 显示区域分布
            district_counts = self.data['district'].value_counts()
            print(f"   - 区域分布:")
            for district, count in district_counts.head(10).items():
                print(f"     {district}: {count}个站点")
            
            return True
            
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            return False
    
    def calculate_station_statistics(self):
        """计算各区站点统计信息"""
        try:
            print("\n📊 正在计算各区站点统计信息...")
            
            # 过滤出有district信息的记录
            valid_data = self.data[
                (self.data['district'].notna()) & 
                (self.data['district'] != '') & 
                (self.data['comprehensive_score'].notna())
            ].copy()
            
            print(f"   - 有效数据: {len(valid_data)}/{len(self.data)} 条")
            
            # 动态计算高分站点阈值（前25%，即75%分位数）
            global HIGH_SCORE_THRESHOLD
            HIGH_SCORE_THRESHOLD = valid_data['comprehensive_score'].quantile(HIGH_SCORE_PERCENTILE / 100)
            high_score_count = len(valid_data[valid_data['comprehensive_score'] >= HIGH_SCORE_THRESHOLD])
            
            print(f"   - 动态计算阈值: {HIGH_SCORE_THRESHOLD:.4f} (前25%分位数)")
            print(f"   - 高分站点总数: {high_score_count}/{len(valid_data)} 个 ({high_score_count/len(valid_data):.1%})")
            
            station_stats = {}
            
            for district in DISTRICT_DATA.keys():
                district_data = valid_data[valid_data['district'] == district]
                
                if len(district_data) == 0:
                    print(f"   ⚠️ {district}: 未找到站点数据")
                    station_stats[district] = {
                        '总站点数': 0,
                        '高分站点数': 0,
                        '平均得分': 0,
                        '高分站点比例': 0
                    }
                    continue
                
                # 基础统计
                total_stations = len(district_data)
                high_score_stations = len(district_data[district_data['comprehensive_score'] >= HIGH_SCORE_THRESHOLD])
                avg_score = district_data['comprehensive_score'].mean()
                high_score_ratio = high_score_stations / total_stations if total_stations > 0 else 0
                
                station_stats[district] = {
                    '总站点数': total_stations,
                    '高分站点数': high_score_stations,
                    '平均得分': avg_score,
                    '高分站点比例': high_score_ratio
                }
                
                print(f"   - {district}: {total_stations}个站点, {high_score_stations}个高分站点 ({high_score_ratio:.2%})")
            
            self.analysis_results['station_stats'] = station_stats
            self.analysis_results['high_score_threshold'] = HIGH_SCORE_THRESHOLD
            return True
            
        except Exception as e:
            print(f"❌ 计算站点统计失败: {e}")
            return False
    
    def calculate_density_indicators(self):
        """计算密度指标"""
        try:
            print("\n🔢 正在计算密度指标...")
            
            density_data = []
            
            for district, stats in self.analysis_results['station_stats'].items():
                if district not in DISTRICT_DATA:
                    continue
                
                area = DISTRICT_DATA[district]['面积']
                population = DISTRICT_DATA[district]['人口数']
                pop_density = DISTRICT_DATA[district]['人口密度']
                aging_rate = AGING_RATE.get(district, 20.0)  # 默认老龄化率20%
                
                total_stations = stats['总站点数']
                high_score_stations = stats['高分站点数']
                avg_score = stats['平均得分']
                high_score_ratio = stats['高分站点比例']
                
                # 计算各种密度指标
                indicators = {
                    '区域': district,
                    '面积': area,
                    '人口数': population,
                    '人口密度': pop_density,
                    '老龄化率': aging_rate,
                    '总站点数': total_stations,
                    '高分站点数': high_score_stations,
                    '平均得分': avg_score,
                    '高分站点比例': high_score_ratio,
                    
                    # 空间密度指标
                    '站点空间密度': total_stations / area if area > 0 else 0,  # 站点数/平方公里
                    '高分站点空间密度': high_score_stations / area if area > 0 else 0,  # 高分站点数/平方公里
                    
                    # 人口密度指标
                    '站点人口密度': (total_stations / population) * 10000 if population > 0 else 0,  # 站点数/万人
                    '高分站点人口密度': (high_score_stations / population) * 10000 if population > 0 else 0,  # 高分站点数/万人
                    
                    # 综合密度指标（考虑老龄化）
                    '老龄化加权站点密度': ((high_score_stations / population) * 10000) * (1 + aging_rate / 100) if population > 0 else 0,
                    
                    # 效率指标
                    '空间利用效率': (high_score_stations / area) / (population / 10000) if area > 0 and population > 0 else 0,
                }
                
                density_data.append(indicators)
                
                print(f"   - {district}:")
                print(f"     高分站点空间密度: {indicators['高分站点空间密度']:.3f} 个/km²")
                print(f"     高分站点人口密度: {indicators['高分站点人口密度']:.3f} 个/万人")
                print(f"     老龄化加权密度: {indicators['老龄化加权站点密度']:.3f}")
            
            self.analysis_results['density_data'] = pd.DataFrame(density_data)
            return True
            
        except Exception as e:
            print(f"❌ 计算密度指标失败: {e}")
            return False
    
    def calculate_zscore_standardization(self):
        """计算Z-score标准化"""
        try:
            print("\n📏 正在进行Z-score标准化...")
            
            df = self.analysis_results['density_data'].copy()
            
            # 需要标准化的指标
            indicators_to_standardize = [
                '站点空间密度', '高分站点空间密度', '站点人口密度', 
                '高分站点人口密度', '老龄化加权站点密度', '空间利用效率',
                '平均得分', '高分站点比例'
            ]
            
            # 计算Z-score标准化
            for indicator in indicators_to_standardize:
                if indicator in df.columns:
                    mean_val = df[indicator].mean()
                    std_val = df[indicator].std()
                    
                    if std_val > 0:
                        df[f'{indicator}_zscore'] = (df[indicator] - mean_val) / std_val
                    else:
                        df[f'{indicator}_zscore'] = 0
                    
                    print(f"   - {indicator}: 均值={mean_val:.3f}, 标准差={std_val:.3f}")
            
            # 计算综合Z-score（多个指标的加权平均）
            weights = {
                '高分站点人口密度_zscore': 0.3,      # 人口服务密度
                '高分站点空间密度_zscore': 0.25,     # 空间覆盖密度
                '老龄化加权站点密度_zscore': 0.2,     # 老龄化考虑
                '平均得分_zscore': 0.15,             # 质量指标
                '空间利用效率_zscore': 0.1           # 效率指标
            }
            
            df['综合密度指数_zscore'] = 0
            for indicator, weight in weights.items():
                if indicator in df.columns:
                    df['综合密度指数_zscore'] += df[indicator] * weight
            
            # 按综合指数排序
            df = df.sort_values('综合密度指数_zscore', ascending=False).reset_index(drop=True)
            df['排名'] = df.index + 1
            
            self.analysis_results['standardized_data'] = df
            
            print("\n🏆 Z-score标准化完成！综合排名前5:")
            for i in range(min(5, len(df))):
                district = df.loc[i, '区域']
                score = df.loc[i, '综合密度指数_zscore']
                print(f"   {i+1}. {district}: {score:.3f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Z-score标准化失败: {e}")
            return False
    
    def save_results(self):
        """保存分析结果"""
        try:
            print("\n💾 正在保存分析结果...")
            
            # 保存详细数据
            detailed_file = self.output_dir / "区域站点密度详细分析.csv"
            self.analysis_results['standardized_data'].to_csv(detailed_file, index=False, encoding='utf-8-sig')
            print(f"✅ 详细分析数据已保存: {detailed_file}")
            
            # 保存排名汇总
            summary_df = self.analysis_results['standardized_data'][
                ['排名', '区域', '总站点数', '高分站点数', '高分站点人口密度', 
                 '高分站点空间密度', '老龄化加权站点密度', '综合密度指数_zscore']
            ].copy()
            
            summary_file = self.output_dir / "区域站点密度排名汇总.csv"
            summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
            print(f"✅ 排名汇总已保存: {summary_file}")
            
            return True
            
        except Exception as e:
            print(f"❌ 保存结果失败: {e}")
            return False
    
    def create_visualizations(self):
        """创建可视化图表"""
        try:
            print("\n📊 正在创建可视化图表...")
            
            df = self.analysis_results['standardized_data']
            
            # 设置图表样式
            plt.style.use('seaborn-v0_8')
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('上海各区高分站点密度分析', fontsize=16, fontweight='bold')
            
            # 1. 综合密度指数排名
            ax1 = axes[0, 0]
            bars1 = ax1.barh(range(len(df)), df['综合密度指数_zscore'], 
                            color=plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(df))))
            ax1.set_yticks(range(len(df)))
            ax1.set_yticklabels(df['区域'])
            ax1.set_xlabel('综合密度指数 (Z-score)')
            ax1.set_title('各区综合密度指数排名')
            ax1.grid(axis='x', alpha=0.3)
            
            # 添加数值标签
            for i, v in enumerate(df['综合密度指数_zscore']):
                ax1.text(v + 0.05 if v >= 0 else v - 0.05, i, f'{v:.2f}', 
                        va='center', ha='left' if v >= 0 else 'right')
            
            # 2. 高分站点人口密度 vs 空间密度
            ax2 = axes[0, 1]
            scatter = ax2.scatter(df['高分站点人口密度'], df['高分站点空间密度'], 
                                 s=df['老龄化率']*10, c=df['综合密度指数_zscore'], 
                                 cmap='RdYlBu_r', alpha=0.7, edgecolors='black')
            ax2.set_xlabel('高分站点人口密度 (个/万人)')
            ax2.set_ylabel('高分站点空间密度 (个/km²)')
            ax2.set_title('人口密度 vs 空间密度\n(气泡大小=老龄化率)')
            
            # 添加区域标签
            for i, district in enumerate(df['区域']):
                ax2.annotate(district, (df.loc[i, '高分站点人口密度'], df.loc[i, '高分站点空间密度']),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            plt.colorbar(scatter, ax=ax2, label='综合密度指数')
            
            # 3. 老龄化率与站点密度关系
            ax3 = axes[1, 0]
            ax3.scatter(df['老龄化率'], df['老龄化加权站点密度'], 
                       c=df['综合密度指数_zscore'], cmap='RdYlBu_r', s=100, alpha=0.7)
            ax3.set_xlabel('老龄化率 (%)')
            ax3.set_ylabel('老龄化加权站点密度')
            ax3.set_title('老龄化率与站点密度关系')
            
            # 添加趋势线
            z = np.polyfit(df['老龄化率'], df['老龄化加权站点密度'], 1)
            p = np.poly1d(z)
            ax3.plot(df['老龄化率'], p(df['老龄化率']), "r--", alpha=0.8)
            
            # 4. 各指标雷达图（前5名）
            ax4 = axes[1, 1]
            top5 = df.head(5)
            
            indicators = ['高分站点人口密度_zscore', '高分站点空间密度_zscore', 
                         '老龄化加权站点密度_zscore', '平均得分_zscore', '空间利用效率_zscore']
            
            angles = np.linspace(0, 2*np.pi, len(indicators), endpoint=False)
            angles = np.concatenate((angles, [angles[0]]))
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(top5)))
            
            for i, (_, row) in enumerate(top5.iterrows()):
                values = [row[ind] for ind in indicators]
                values += [values[0]]  # 闭合
                
                ax4.plot(angles, values, 'o-', linewidth=2, 
                        label=row['区域'], color=colors[i])
                ax4.fill(angles, values, alpha=0.1, color=colors[i])
            
            ax4.set_xticks(angles[:-1])
            ax4.set_xticklabels(['人口密度', '空间密度', '老龄化加权', '平均得分', '空间效率'])
            ax4.set_title('前5名区域各指标雷达图')
            ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax4.grid(True)
            
            plt.tight_layout()
            
            # 保存图表
            chart_file = self.output_dir / "区域站点密度分析图表.png"
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            print(f"✅ 可视化图表已保存: {chart_file}")
            
            plt.show()
            
            return True
            
        except Exception as e:
            print(f"❌ 创建可视化失败: {e}")
            return False
    
    def generate_report(self):
        """生成分析报告"""
        try:
            print("\n📝 正在生成分析报告...")
            
            df = self.analysis_results['standardized_data']
            
            report = []
            report.append("# 上海各区高分站点密度分析报告")
            report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report.append("")
            
            # 1. 执行摘要
            report.append("## 一、执行摘要")
            report.append(f"本分析基于{len(self.data)}个公交站点数据，使用Z-score标准化方法")
            report.append(f"对上海{len(df)}个区的高分站点密度进行跨区比较。")
            
            # 获取动态计算的阈值
            threshold = self.analysis_results.get('high_score_threshold', HIGH_SCORE_THRESHOLD)
            report.append(f"高分站点定义为综合得分排名前25%的站点（阈值≥{threshold:.4f}）。")
            report.append("")
            
            # 2. 排名结果
            report.append("## 二、综合密度指数排名")
            report.append("| 排名 | 区域 | 高分站点数 | 人口密度(个/万人) | 空间密度(个/km²) | 综合指数 |")
            report.append("|------|------|------------|-------------------|------------------|----------|")
            
            for _, row in df.head(10).iterrows():
                report.append(f"| {row['排名']} | {row['区域']} | {row['高分站点数']} | "
                            f"{row['高分站点人口密度']:.3f} | {row['高分站点空间密度']:.3f} | "
                            f"{row['综合密度指数_zscore']:.3f} |")
            
            report.append("")
            
            # 3. 主要发现
            report.append("## 三、主要发现")
            
            top3 = df.head(3)['区域'].tolist()
            bottom3 = df.tail(3)['区域'].tolist()
            
            report.append(f"### 3.1 优势区域")
            report.append(f"综合密度指数排名前三的区域为：{', '.join(top3)}")
            
            # 找出各单项指标的最优区域
            pop_best = df.loc[df['高分站点人口密度'].idxmax(), '区域']
            space_best = df.loc[df['高分站点空间密度'].idxmax(), '区域']
            aging_best = df.loc[df['老龄化加权站点密度'].idxmax(), '区域']
            
            report.append(f"- 人口服务密度最优：{pop_best}")
            report.append(f"- 空间覆盖密度最优：{space_best}")
            report.append(f"- 老龄化适配最优：{aging_best}")
            report.append("")
            
            report.append(f"### 3.2 待改善区域")
            report.append(f"综合密度指数排名后三的区域为：{', '.join(bottom3)}")
            report.append("建议重点关注这些区域的公交站点建设和质量提升。")
            report.append("")
            
            # 4. 相关性分析
            report.append("## 四、相关性分析")
            
            # 计算相关系数
            corr_aging = df['老龄化率'].corr(df['老龄化加权站点密度'])
            corr_pop_space = df['高分站点人口密度'].corr(df['高分站点空间密度'])
            
            report.append(f"- 老龄化率与站点密度相关性：{corr_aging:.3f}")
            report.append(f"- 人口密度与空间密度相关性：{corr_pop_space:.3f}")
            report.append("")
            
            # 5. 建议
            report.append("## 五、政策建议")
            report.append("1. **优化资源配置**：向密度指数较低的区域倾斜资源")
            report.append("2. **考虑人口结构**：老龄化率高的区域需要更密集的高质量站点")
            report.append("3. **平衡发展**：在保证中心城区服务质量的同时，提升郊区覆盖密度")
            report.append("4. **持续监测**：建立定期评估机制，动态调整站点布局")
            report.append("")
            
            # 保存报告
            report_file = self.output_dir / "区域站点密度分析报告.md"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report))
            
            print(f"✅ 分析报告已保存: {report_file}")
            
            # 显示核心结果
            print("\n🏆 核心分析结果:")
            print("="*50)
            for i in range(min(5, len(df))):
                row = df.iloc[i]
                print(f"{i+1}. {row['区域']}: 综合指数 {row['综合密度指数_zscore']:.3f}")
                print(f"   高分站点 {row['高分站点数']}个 | "
                      f"人口密度 {row['高分站点人口密度']:.3f}/万人 | "
                      f"空间密度 {row['高分站点空间密度']:.3f}/km²")
            
            return True
            
        except Exception as e:
            print(f"❌ 生成报告失败: {e}")
            return False
    
    def run_analysis(self):
        """运行完整分析"""
        try:
            print("🚀 开始区域站点密度分析...")
            print("="*60)
            
            # 1. 加载数据
            if not self.load_data():
                return False
            
            # 2. 计算站点统计
            if not self.calculate_station_statistics():
                return False
            
            # 3. 计算密度指标
            if not self.calculate_density_indicators():
                return False
            
            # 4. Z-score标准化
            if not self.calculate_zscore_standardization():
                return False
            
            # 5. 保存结果
            if not self.save_results():
                return False
            
            # 6. 创建可视化
            if not self.create_visualizations():
                return False
            
            # 7. 生成报告
            if not self.generate_report():
                return False
            
            print("\n" + "="*60)
            print("🎉 区域站点密度分析完成！")
            print("="*60)
            
            return True
            
        except Exception as e:
            print(f"❌ 分析失败: {e}")
            return False

def main():
    """主函数"""
    print("🚀 上海各区高分站点密度分析程序")
    print("结合人口老龄化率和空间密度，使用Z-score标准化进行跨区比较")
    print("="*60)
    
    # 检查输入文件
    if not Path(INPUT_CSV).exists():
        print(f"❌ 错误: 输入文件不存在: {INPUT_CSV}")
        print("请确保final_station_ranking_complete.csv文件存在")
        return
    
    # 创建分析器
    analyzer = DistrictDensityAnalyzer(INPUT_CSV, OUTPUT_DIR)
    
    # 运行分析
    success = analyzer.run_analysis()
    
    if success:
        print(f"\n输出文件:")
        print(f"  1. 区域站点密度详细分析.csv - 完整分析数据")
        print(f"  2. 区域站点密度排名汇总.csv - 排名汇总")
        print(f"  3. 区域站点密度分析图表.png - 可视化图表")
        print(f"  4. 区域站点密度分析报告.md - 分析报告")
        print(f"\n所有文件已保存到: {OUTPUT_DIR}")
    else:
        print("\n❌ 分析失败，请查看错误信息")

if __name__ == "__main__":
    main() 