#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
老年人口与公交站点适老化程度关系分析
基于2020年人口普查数据，分析各区老年人口分布与站点适老化程度的相关性
将各区分为四象限：高需求-高供给、高需求-低供给、低需求-高供给、低需求-低供给
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 配置区域 ====================
# 输入文件配置
INPUT_CSV = "/Users/songdingan/Downloads/detect_bus/final_station_ranking_complete.csv"
OUTPUT_DIR = "/Users/songdingan/Downloads/detect_bus/"

# 2020年人口普查65岁以上人口数据
ELDERLY_POPULATION_DATA = {
    '黄浦区': {'65岁以上人口数': 122373, '65岁以上人口占比': 18.48},
    '徐汇区': {'65岁以上人口数': 229388, '65岁以上人口占比': 20.61},
    '长宁区': {'65岁以上人口数': 143130, '65岁以上人口占比': 20.65},
    '静安区': {'65岁以上人口数': 214376, '65岁以上人口占比': 21.97},
    '普陀区': {'65岁以上人口数': 261901, '65岁以上人口占比': 21.12},
    '虹口区': {'65岁以上人口数': 175984, '65岁以上人口占比': 23.23},
    '杨浦区': {'65岁以上人口数': 271556, '65岁以上人口占比': 21.85},
    '闵行区': {'65岁以上人口数': 366575, '65岁以上人口占比': 14.02},
    '宝山区': {'65岁以上人口数': 301248, '65岁以上人口占比': 15.85},
    '嘉定区': {'65岁以上人口数': 148682, '65岁以上人口占比': 12.8},
    '浦东新区': {'65岁以上人口数': 694395, '65岁以上人口占比': 15.03},
    '金山区': {'65岁以上人口数': 34604, '65岁以上人口占比': 12.37},
    '松江区': {'65岁以上人口数': 126986, '65岁以上人口占比': 9.74},
    '青浦区': {'65岁以上人口数': 75644, '65岁以上人口占比': 9.3},
    '奉贤区': {'65岁以上人口数': 36891, '65岁以上人口占比': 9.79}
}

# 区域基础数据
DISTRICT_DATA = {
    '黄浦区': {'面积': 20.52, '总人口': 662030},
    '徐汇区': {'面积': 54.93, '总人口': 1113078},
    '长宁区': {'面积': 38.3, '总人口': 693051},
    '静安区': {'面积': 37.37, '总人口': 975707},
    '普陀区': {'面积': 55.53, '总人口': 1239800},
    '虹口区': {'面积': 23.45, '总人口': 757498},
    '杨浦区': {'面积': 60.61, '总人口': 1242548},
    '闵行区': {'面积': 372.56, '总人口': 2614218},
    '宝山区': {'面积': 271.3, '总人口': 1900809},
    '嘉定区': {'面积': 463.55, '总人口': 1161435},
    '浦东新区': {'面积': 1210.41, '总人口': 4619851},
    '金山区': {'面积': 613.28, '总人口': 279796},
    '松江区': {'面积': 604.67, '总人口': 1304271},
    '青浦区': {'面积': 676.26, '总人口': 813128},
    '奉贤区': {'面积': 687.39, '总人口': 376669}
}

# 适老化设施权重（基于重要性）
AGING_FRIENDLY_WEIGHTS = {
    'zy_ratio': 0.25,    # 座椅比例
    'lg_ratio': 0.20,    # 栏杆比例
    'dp_ratio': 0.15,    # 顶棚比例
    'fs_ratio': 0.15,    # 扶手比例
    'cbfh_ratio': 0.10,  # 侧边防护比例
    'zp_ratio': 0.10,    # 站牌比例
    'comprehensive_score': 0.05  # 综合得分权重
}
# ==================================================

class ElderlyAdaptabilityAnalyzer:
    """老年人口与适老化程度关系分析器"""
    
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
            
            print(f"✅ 数据加载成功!")
            print(f"   - 总站点数: {len(self.data)}")
            print(f"   - 列名: {list(self.data.columns)}")
            
            return True
            
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            return False
    
    def calculate_aging_adaptability_index(self):
        """计算各区适老化指数"""
        try:
            print("\n🏥 正在计算各区适老化指数...")
            
            # 过滤有效数据
            valid_data = self.data[
                (self.data['district'].notna()) & 
                (self.data['district'] != '') & 
                (self.data['comprehensive_score'].notna())
            ].copy()
            
            adaptability_data = []
            
            for district in DISTRICT_DATA.keys():
                district_data = valid_data[valid_data['district'] == district]
                
                if len(district_data) == 0:
                    print(f"   ⚠️ {district}: 未找到站点数据")
                    continue
                
                # 计算各类设施的覆盖率
                total_stations = len(district_data)
                
                # 检查各类设施列是否存在
                facility_ratios = {}
                facility_counts = {}
                
                # 适老化设施类型映射
                facility_mapping = {
                    'zy_ratio': ['zy_count', 'ZY'],  # 座椅
                    'lg_ratio': ['lg_count', 'LG'],  # 栏杆  
                    'dp_ratio': ['dp_count', 'DP'],  # 顶棚
                    'fs_ratio': ['fs_count', 'FS'],  # 扶手
                    'cbfh_ratio': ['cbfh_count', 'CBFH'],  # 侧边防护
                    'zp_ratio': ['zp_count', 'ZP']   # 站牌
                }
                
                for ratio_key, count_keys in facility_mapping.items():
                    # 尝试找到对应的计数列
                    count = 0
                    for col_key in count_keys:
                        if col_key in district_data.columns:
                            count = district_data[col_key].sum()
                            break
                    
                    facility_counts[ratio_key.replace('_ratio', '_count')] = count
                    facility_ratios[ratio_key] = count / total_stations if total_stations > 0 else 0
                
                # 计算综合适老化指数
                adaptability_index = 0
                for ratio_key, weight in AGING_FRIENDLY_WEIGHTS.items():
                    if ratio_key == 'comprehensive_score':
                        # 使用平均综合得分
                        avg_score = district_data['comprehensive_score'].mean()
                        adaptability_index += avg_score * weight
                    elif ratio_key in facility_ratios:
                        adaptability_index += facility_ratios[ratio_key] * weight
                
                # 获取老年人口数据
                elderly_data = ELDERLY_POPULATION_DATA.get(district, {})
                elderly_population = elderly_data.get('65岁以上人口数', 0)
                elderly_ratio = elderly_data.get('65岁以上人口占比', 0)
                
                # 计算老年人口密度指标
                area = DISTRICT_DATA[district]['面积']
                elderly_density = elderly_population / area if area > 0 else 0  # 老年人口/平方公里
                elderly_per_10k = (elderly_population / DISTRICT_DATA[district]['总人口']) * 10000 if DISTRICT_DATA[district]['总人口'] > 0 else 0
                
                district_result = {
                    '区域': district,
                    '总站点数': total_stations,
                    '老年人口数': elderly_population,
                    '老年人口占比': elderly_ratio,
                    '老年人口密度': elderly_density,
                    '适老化指数': adaptability_index,
                    '平均综合得分': district_data['comprehensive_score'].mean(),
                    **facility_ratios,
                    **facility_counts
                }
                
                adaptability_data.append(district_result)
                
                print(f"   - {district}: 老年人口{elderly_population:,}人 ({elderly_ratio:.1f}%), "
                      f"适老化指数{adaptability_index:.3f}")
            
            self.analysis_results['adaptability_data'] = pd.DataFrame(adaptability_data)
            return True
            
        except Exception as e:
            print(f"❌ 计算适老化指数失败: {e}")
            return False
    
    def calculate_correlation_analysis(self):
        """计算相关性分析"""
        try:
            print("\n📊 正在进行相关性分析...")
            
            df = self.analysis_results['adaptability_data']
            
            if len(df) < 3:
                print("   ⚠️ 数据点太少，无法进行可靠的相关性分析")
                return False
            
            # 计算皮尔森相关系数
            correlations = {}
            
            # 老年人口数量 vs 适老化指数
            elderly_pop = df['老年人口数'].values
            adaptability = df['适老化指数'].values
            
            r_population, p_population = pearsonr(elderly_pop, adaptability)
            correlations['老年人口数_适老化指数'] = {'相关系数': r_population, 'p值': p_population}
            
            # 老年人口占比 vs 适老化指数
            elderly_ratio = df['老年人口占比'].values
            r_ratio, p_ratio = pearsonr(elderly_ratio, adaptability)
            correlations['老年人口占比_适老化指数'] = {'相关系数': r_ratio, 'p值': p_ratio}
            
            # 老年人口密度 vs 适老化指数
            elderly_density = df['老年人口密度'].values
            r_density, p_density = pearsonr(elderly_density, adaptability)
            correlations['老年人口密度_适老化指数'] = {'相关系数': r_density, 'p值': p_density}
            
            self.analysis_results['correlations'] = correlations
            
            print(f"   📈 相关性分析结果:")
            for key, value in correlations.items():
                r = value['相关系数']
                p = value['p值']
                significance = "显著" if p < 0.05 else "不显著"
                
                if abs(r) >= 0.5:
                    strength = "强相关"
                elif abs(r) >= 0.3:
                    strength = "中等相关"
                else:
                    strength = "弱相关"
                
                direction = "正相关" if r > 0 else "负相关"
                
                print(f"     {key}: r={r:.3f} ({direction}, {strength}, {significance})")
            
            return True
            
        except Exception as e:
            print(f"❌ 相关性分析失败: {e}")
            return False
    
    def four_quadrant_classification(self):
        """四象限分类"""
        try:
            print("\n🔄 正在进行四象限分类...")
            
            df = self.analysis_results['adaptability_data'].copy()
            
            # 计算中位数作为分界线
            elderly_median = df['老年人口数'].median()
            adaptability_median = df['适老化指数'].median()
            
            print(f"   - 老年人口数中位数: {elderly_median:,.0f}人")
            print(f"   - 适老化指数中位数: {adaptability_median:.3f}")
            
            # 分类函数
            def classify_district(row):
                elderly_high = row['老年人口数'] > elderly_median
                adaptability_high = row['适老化指数'] > adaptability_median
                
                if elderly_high and adaptability_high:
                    return "Type I (High Demand - High Supply)"
                elif elderly_high and not adaptability_high:
                    return "Type II (High Demand - Low Supply)"
                elif not elderly_high and adaptability_high:
                    return "Type III (Low Demand - High Supply)"
                else:
                    return "Type IV (Low Demand - Low Supply)"
            
            # 应用分类
            df['象限分类'] = df.apply(classify_district, axis=1)
            
            # 统计各类区域
            classification_stats = df['象限分类'].value_counts()
            
            print(f"\n   🏷️ 四象限分类结果:")
            for category, count in classification_stats.items():
                percentage = count / len(df) * 100
                print(f"     {category}: {count}个区域 ({percentage:.1f}%)")
            
            # 详细展示每个象限的区域
            print(f"\n   📍 各象限具体区域:")
            for category in ["Type I (High Demand - High Supply)", "Type II (High Demand - Low Supply)", 
                            "Type III (Low Demand - High Supply)", "Type IV (Low Demand - Low Supply)"]:
                districts = df[df['象限分类'] == category]['区域'].tolist()
                if districts:
                    print(f"     {category}: {', '.join(districts)}")
            
            self.analysis_results['classified_data'] = df
            self.analysis_results['classification_stats'] = classification_stats
            self.analysis_results['thresholds'] = {
                '老年人口数中位数': elderly_median,
                '适老化指数中位数': adaptability_median
            }
            
            return True
            
        except Exception as e:
            print(f"❌ 四象限分类失败: {e}")
            return False
    
    def create_visualizations(self):
        """创建可视化图表"""
        try:
            print("\n📊 正在创建可视化图表...")
            
            df = self.analysis_results['classified_data']
            correlations = self.analysis_results['correlations']
            
            # 创建2x2的子图
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Elderly Population vs Bus Stop Age-Friendly Facilities Analysis', fontsize=16, fontweight='bold')
            
            # 颜色映射
            category_colors = {
                "Type I (High Demand - High Supply)": '#2E8B57',      # 深绿色 - 理想状态
                "Type II (High Demand - Low Supply)": '#DC143C',      # 深红色 - 需要改善
                "Type III (Low Demand - High Supply)": '#4169E1',     # 蓝色 - 资源富余
                "Type IV (Low Demand - Low Supply)": '#FFA500'        # 橙色 - 优先级较低
            }
            
            # 1. 四象限散点图
            ax1 = axes[0, 0]
            for category in category_colors.keys():
                category_data = df[df['象限分类'] == category]
                if len(category_data) > 0:
                    ax1.scatter(category_data['老年人口数'], category_data['适老化指数'], 
                              c=category_colors[category], label=category, s=100, alpha=0.7)
                    
                    # 添加区域标签
                    for _, row in category_data.iterrows():
                        ax1.annotate(row['区域'], 
                                   (row['老年人口数'], row['适老化指数']),
                                   xytext=(5, 5), textcoords='offset points', 
                                   fontsize=8, alpha=0.8)
            
            # 添加中位数线
            ax1.axvline(x=self.analysis_results['thresholds']['老年人口数中位数'], 
                       color='gray', linestyle='--', alpha=0.5, label='Elderly Population Median')
            ax1.axhline(y=self.analysis_results['thresholds']['适老化指数中位数'], 
                       color='gray', linestyle='--', alpha=0.5, label='Age-Friendly Index Median')
            
            ax1.set_xlabel('Elderly Population (persons)')
            ax1.set_ylabel('Age-Friendly Index')
            ax1.set_title('Four-Quadrant Classification')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.grid(True, alpha=0.3)
            
            # 2. 老年人口占比 vs 适老化指数
            ax2 = axes[0, 1]
            colors = [category_colors[cat] for cat in df['象限分类']]
            scatter = ax2.scatter(df['老年人口占比'], df['适老化指数'], 
                                c=colors, s=100, alpha=0.7)
            
            # 添加趋势线
            z = np.polyfit(df['老年人口占比'], df['适老化指数'], 1)
            p = np.poly1d(z)
            ax2.plot(df['老年人口占比'], p(df['老年人口占比']), "r--", alpha=0.8)
            
            # 添加相关系数
            r = correlations['老年人口占比_适老化指数']['相关系数']
            ax2.text(0.05, 0.95, f'Pearson Correlation: r={r:.3f}', 
                    transform=ax2.transAxes, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
            
            ax2.set_xlabel('Elderly Population Percentage (%)')
            ax2.set_ylabel('Age-Friendly Index')
            ax2.set_title('Elderly Population % vs Age-Friendly Index')
            ax2.grid(True, alpha=0.3)
            
            # 3. 各象限区域数量统计
            ax3 = axes[1, 0]
            stats = self.analysis_results['classification_stats']
            bars = ax3.bar(range(len(stats)), stats.values, 
                          color=[category_colors[cat] for cat in stats.index])
            
            ax3.set_xticks(range(len(stats)))
            ax3.set_xticklabels([cat.split('(')[0] for cat in stats.index], rotation=45)
            ax3.set_ylabel('Number of Districts')
            ax3.set_title('Distribution by Quadrant')
            
            # 添加数值标签
            for i, v in enumerate(stats.values):
                ax3.text(i, v + 0.1, str(v), ha='center', va='bottom')
            
            # 4. 适老化指数排名
            ax4 = axes[1, 1]
            df_sorted = df.sort_values('适老化指数', ascending=True)
            colors_sorted = [category_colors[cat] for cat in df_sorted['象限分类']]
            
            bars4 = ax4.barh(range(len(df_sorted)), df_sorted['适老化指数'], 
                           color=colors_sorted, alpha=0.7)
            ax4.set_yticks(range(len(df_sorted)))
            ax4.set_yticklabels(df_sorted['区域'])
            ax4.set_xlabel('Age-Friendly Index')
            ax4.set_title('Age-Friendly Index Ranking by District')
            ax4.grid(axis='x', alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图表
            chart_file = self.output_dir / "elderly_population_analysis_charts.png"
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            print(f"✅ 可视化图表已保存: {chart_file}")
            
            plt.show()
            
            return True
            
        except Exception as e:
            print(f"❌ 创建可视化失败: {e}")
            return False
    
    def generate_comprehensive_report(self):
        """生成综合分析报告"""
        try:
            print("\n📝 正在生成综合分析报告...")
            
            df = self.analysis_results['classified_data']
            correlations = self.analysis_results['correlations']
            stats = self.analysis_results['classification_stats']
            
            report = []
            report.append("# 上海各区老年人口与公交站点适老化程度关系分析报告")
            report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report.append("")
            
            # 1. 执行摘要
            report.append("## 一、执行摘要")
            report.append(f"本分析基于2020年人口普查数据和{len(self.data)}个公交站点数据，")
            report.append(f"对上海{len(df)}个区的老年人口分布与公交站点适老化程度进行关联分析。")
            
            # 获取主要相关系数
            main_correlation = correlations['老年人口数_适老化指数']['相关系数']
            ratio_correlation = correlations['老年人口占比_适老化指数']['相关系数']
            
            if abs(main_correlation) >= 0.5:
                corr_strength = "强相关"
            elif abs(main_correlation) >= 0.3:
                corr_strength = "中等相关"
            else:
                corr_strength = "弱相关"
            
            corr_direction = "正相关" if main_correlation > 0 else "负相关"
            
            report.append(f"主要发现：老年人口数量与适老化指数呈{corr_direction}（r={main_correlation:.3f}，{corr_strength}）。")
            report.append("")
            
            # 2. 相关性分析结果
            report.append("## 二、相关性分析结果")
            report.append("### 2.1 皮尔森相关系数分析")
            report.append("| Indicator Combination | Correlation(r) | Strength | Significance |")
            report.append("|----------------------|----------------|----------|--------------|")
            
            for key, value in correlations.items():
                r = value['相关系数']
                p = value['p值']
                
                if abs(r) >= 0.5:
                    strength = "强相关"
                elif abs(r) >= 0.3:
                    strength = "中等相关"
                else:
                    strength = "弱相关"
                
                significance = "显著(p<0.05)" if p < 0.05 else "不显著(p≥0.05)"
                direction = "正" if r > 0 else "负"
                
                report.append(f"| {key} | {r:.3f}({direction}) | {strength} | {significance} |")
            
            report.append("")
            
            # 3. 四象限分类结果
            report.append("## 三、四象限分类结果")
            report.append("### 3.1 分类标准")
            elderly_median = self.analysis_results['thresholds']['老年人口数中位数']
            adaptability_median = self.analysis_results['thresholds']['适老化指数中位数']
            
            report.append(f"- 老年人口数分界线：{elderly_median:,.0f}人（中位数）")
            report.append(f"- 适老化指数分界线：{adaptability_median:.3f}（中位数）")
            report.append("")
            
            report.append("### 3.2 各象限统计")
            report.append("| Quadrant Type | District Count | Percentage | Districts |")
            report.append("|---------------|----------------|------------|-----------|")
            
            for category in ["Type I (High Demand - High Supply)", "Type II (High Demand - Low Supply)", 
                            "Type III (Low Demand - High Supply)", "Type IV (Low Demand - Low Supply)"]:
                count = stats.get(category, 0)
                percentage = count / len(df) * 100 if len(df) > 0 else 0
                districts = df[df['象限分类'] == category]['区域'].tolist()
                districts_str = ', '.join(districts) if districts else "None"
                
                report.append(f"| {category} | {count} | {percentage:.1f}% | {districts_str} |")
            
            report.append("")
            
            # 4. 关键发现与解读
            report.append("## 四、关键发现与解读")
            
            # 4.1 相关性解读
            report.append("### 4.1 相关性分析解读")
            if abs(main_correlation) >= 0.5:
                if main_correlation > 0:
                    report.append("**强正相关**：老年人口多的区域，适老化程度也较高，资源配置相对合理。")
                else:
                    report.append("**强负相关**：老年人口多的区域，适老化程度反而较低，存在严重的供需错配。")
            else:
                report.append("**弱相关性**：老年人口分布与适老化程度无明显关联，资源配置存在结构性问题。")
            
            report.append("")
            
            # 4.2 象限分析
            report.append("### 4.2 各象限特征分析")
            
            # Type II区域（重点关注）
            type2_districts = df[df['象限分类'] == "Type II (High Demand - Low Supply)"]['区域'].tolist()
            type2_count = len(type2_districts)
            
            if type2_count > 0:
                report.append(f"**Type II Districts (High Demand - Low Supply)**：{type2_count} districts including {', '.join(type2_districts)}.")
                report.append("These districts have large elderly populations but insufficient age-friendly facilities, requiring priority improvement.")
            
            # Type III区域（资源优化）
            type3_districts = df[df['象限分类'] == "Type III (Low Demand - High Supply)"]['区域'].tolist()
            type3_count = len(type3_districts)
            
            if type3_count > 0:
                report.append(f"**Type III Districts (Low Demand - High Supply)**：{type3_count} districts including {', '.join(type3_districts)}.")
                report.append("These districts have good age-friendly facilities but relatively fewer elderly population, resource reallocation may be considered.")
            
            report.append("")
            
            # 5. 政策建议
            report.append("## 五、政策建议")
            report.append("### 5.1 优先改造区域")
            if type2_count > 0:
                report.append(f"重点关注Ⅱ类区域（{', '.join(type2_districts)}），优先进行适老化改造：")
                report.append("- 增加无障碍设施（栏杆、扶手）")
                report.append("- 完善候车环境（座椅、顶棚）")
                report.append("- 提升站点安全性（侧边防护）")
            
            report.append("")
            
            report.append("### 5.2 资源优化调配")
            if type3_count > 0:
                report.append(f"对于Ⅲ类区域（{', '.join(type3_districts)}），考虑：")
                report.append("- 经验输出：将成功做法推广到其他区域")
                report.append("- 资源调配：适当调整资源向更需要的区域倾斜")
            
            report.append("")
            
            report.append("### 5.3 综合发展策略")
            report.append("1. **建立动态评估机制**：定期更新老年人口数据和适老化指数")
            report.append("2. **因地制宜改造**：根据各区实际情况制定差异化改造方案")
            report.append("3. **跨区域协调**：促进优势区域与待改善区域的经验交流")
            report.append("4. **长期规划导向**：考虑人口老龄化趋势，前瞻性布局适老化设施")
            
            # 6. 详细数据表
            report.append("## 六、详细数据表")
            report.append("| District | Elderly Population | Elderly % | Age-Friendly Index | Quadrant |")
            report.append("|----------|--------------------|-----------|--------------------|----------|")
            
            for _, row in df.sort_values('适老化指数', ascending=False).iterrows():
                report.append(f"| {row['区域']} | {row['老年人口数']:,} | {row['老年人口占比']:.1f} | "
                            f"{row['适老化指数']:.3f} | {row['象限分类']} |")
            
            # 保存报告
            report_file = self.output_dir / "elderly_population_analysis_report.md"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report))
            
            print(f"✅ 综合分析报告已保存: {report_file}")
            
            return True
            
        except Exception as e:
            print(f"❌ 生成报告失败: {e}")
            return False
    
    def save_results(self):
        """保存分析结果"""
        try:
            print("\n💾 正在保存分析结果...")
            
            # 保存详细数据
            detailed_file = self.output_dir / "elderly_population_detailed_analysis.csv"
            self.analysis_results['classified_data'].to_csv(detailed_file, index=False, encoding='utf-8-sig')
            print(f"✅ 详细分析数据已保存: {detailed_file}")
            
            # 保存相关性分析结果
            corr_data = []
            for key, value in self.analysis_results['correlations'].items():
                corr_data.append({
                    'Indicator_Combination': key,
                    'Correlation_Coefficient': value['相关系数'],
                    'P_Value': value['p值'],
                    'Significance': 'Yes' if value['p值'] < 0.05 else 'No'
                })
            
            corr_file = self.output_dir / "correlation_analysis_results.csv"
            pd.DataFrame(corr_data).to_csv(corr_file, index=False, encoding='utf-8-sig')
            print(f"✅ 相关性分析结果已保存: {corr_file}")
            
            # 保存分类统计
            stats_data = []
            for category, count in self.analysis_results['classification_stats'].items():
                percentage = count / len(self.analysis_results['classified_data']) * 100
                districts = self.analysis_results['classified_data'][
                    self.analysis_results['classified_data']['象限分类'] == category
                ]['区域'].tolist()
                
                stats_data.append({
                    'Quadrant_Type': category,
                    'District_Count': count,
                    'Percentage': percentage,
                    'Districts': ', '.join(districts)
                })
            
            stats_file = self.output_dir / "quadrant_classification_statistics.csv"
            pd.DataFrame(stats_data).to_csv(stats_file, index=False, encoding='utf-8-sig')
            print(f"✅ 四象限分类统计已保存: {stats_file}")
            
            return True
            
        except Exception as e:
            print(f"❌ 保存结果失败: {e}")
            return False
    
    def run_analysis(self):
        """运行完整分析"""
        try:
            print("🚀 开始老年人口与适老化程度关系分析...")
            print("="*60)
            
            # 1. 加载数据
            if not self.load_data():
                return False
            
            # 2. 计算适老化指数
            if not self.calculate_aging_adaptability_index():
                return False
            
            # 3. 相关性分析
            if not self.calculate_correlation_analysis():
                return False
            
            # 4. 四象限分类
            if not self.four_quadrant_classification():
                return False
            
            # 5. 创建可视化
            if not self.create_visualizations():
                return False
            
            # 6. 保存结果
            if not self.save_results():
                return False
            
            # 7. 生成报告
            if not self.generate_comprehensive_report():
                return False
            
            print("\n" + "="*60)
            print("🎉 老年人口与适老化程度关系分析完成！")
            
            # 显示核心结果
            df = self.analysis_results['classified_data']
            correlations = self.analysis_results['correlations']
            
            print("\n📊 核心分析结果:")
            print("="*50)
            
            # 相关性结果
            main_corr = correlations['老年人口数_适老化指数']['相关系数']
            print(f"🔗 老年人口数与适老化指数相关性: r = {main_corr:.3f}")
            
            if abs(main_corr) >= 0.5:
                print("   → 强相关：供需匹配度较好" if main_corr > 0 else "   → 强负相关：存在严重供需错配")
            else:
                print("   → 弱相关：资源配置需要优化")
            
            # 分类结果
            stats = self.analysis_results['classification_stats']
            type2_count = stats.get("Type II (High Demand - Low Supply)", 0)
            type3_count = stats.get("Type III (Low Demand - High Supply)", 0)
            
            print(f"\n🏷️ 四象限分类结果:")
            print(f"   Type II (High Demand - Low Supply): {type2_count}个区域 - Priority Renovation")
            print(f"   Type III (Low Demand - High Supply): {type3_count}个区域 - Resource Optimization")
            
            # 重点关注区域
            if type2_count > 0:
                type2_districts = df[df['象限分类'] == "Type II (High Demand - Low Supply)"]['区域'].tolist()
                print(f"\n🚨 Priority Renovation Districts: {', '.join(type2_districts)}")
            
            print("="*50)
            
            return True
            
        except Exception as e:
            print(f"❌ 分析失败: {e}")
            return False

def main():
    """主函数"""
    print("🚀 上海各区老年人口与公交站点适老化程度关系分析")
    print("基于2020年人口普查数据，分析老年人口分布与适老化程度的相关性")
    print("="*60)
    
    # 检查输入文件
    if not Path(INPUT_CSV).exists():
        print(f"❌ 错误: 输入文件不存在: {INPUT_CSV}")
        print("请确保final_station_ranking_complete.csv文件存在")
        return
    
    # 创建分析器
    analyzer = ElderlyAdaptabilityAnalyzer(INPUT_CSV, OUTPUT_DIR)
    
    # 运行分析
    success = analyzer.run_analysis()
    
    if success:
        print(f"\n📁 Output Files:")
        print(f"  1. elderly_population_detailed_analysis.csv - Complete analysis data")
        print(f"  2. correlation_analysis_results.csv - Pearson correlation coefficients")
        print(f"  3. quadrant_classification_statistics.csv - Classification statistics")
        print(f"  4. elderly_population_analysis_charts.png - Visualization charts")
        print(f"  5. elderly_population_analysis_report.md - Comprehensive analysis report")
        print(f"\nAll files saved to: {OUTPUT_DIR}")
    else:
        print("\n❌ Analysis failed, please check error messages")

if __name__ == "__main__":
    main() 