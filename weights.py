#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
熵权法+蒙特卡洛模拟优化权重计算程序
用于计算子指标权重、一级维度得分，并通过蒙特卡洛模拟找到最稳健的权重组合
"""

import numpy as np
import pandas as pd
from pathlib import Path
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置区域 ====================
# 输入输出配置
INPUT_CSV = "/Users/songdingan/Downloads/detect_bus/reg_results.csv"           # ← 输入CSV文件路径
OUTPUT_DIR = "/Users/songdingan/Downloads/detect_bus"  # ← 输出目录路径

# 蒙特卡洛模拟配置
MONTE_CARLO_ITERATIONS = 10000                # 蒙特卡洛模拟次数
RANDOM_SEED = 42                             # 随机种子，确保结果可重现

# 指标配置（根据实际数据调整）
# 定义一级维度和对应的二级指标
INDICATOR_HIERARCHY = {
    'Safety and Comfort in Waiting Areas': ['侧边防护', '顶棚', '座椅扶手', '栏杆', '座椅', '缓冲区'],  # 等候区安全舒适性
    'Information Accessibility': ['站牌', '电子站牌'],                                              # 信息可达性
    'Barrier-free Access': ['盲道']                                                               # 无障碍通行
}

# 一级维度权重约束（可选，如果不设置则等权重）
PRIMARY_WEIGHT_CONSTRAINTS = {
    'Safety and Comfort in Waiting Areas': (0.4, 0.7),    # 最小40%，最大70% - 最重要的维度
    'Information Accessibility': (0.2, 0.4),               # 最小20%，最大40% - 信息服务
    'Barrier-free Access': (0.1, 0.2)                     # 最小10%，最大20% - 无障碍设施
}
# ==================================================

class EntropyWeightCalculator:
    """熵权法权重计算器"""
    
    def __init__(self, input_csv, output_dir, indicator_hierarchy):
        self.input_csv = input_csv
        self.output_dir = Path(output_dir)
        self.indicator_hierarchy = indicator_hierarchy
        self.data = None
        self.results = {}
        
    def load_data(self):
        """加载数据"""
        try:
            print(f"📂 正在加载数据: {self.input_csv}")
            self.data = pd.read_csv(self.input_csv)
            
            # 提取指标列（排除非指标列）
            non_indicator_cols = ['filename', 'image_id', 'longitude', 'latitude', 'time_code']
            self.indicator_cols = [col for col in self.data.columns if col not in non_indicator_cols]
            
            # 验证指标列是否与配置一致
            expected_indicators = []
            for indicators in INDICATOR_HIERARCHY.values():
                expected_indicators.extend(indicators)
            
            missing_indicators = set(expected_indicators) - set(self.indicator_cols)
            if missing_indicators:
                print(f"⚠️ 警告: 配置中的指标在数据中缺失: {missing_indicators}")
            
            extra_indicators = set(self.indicator_cols) - set(expected_indicators)
            if extra_indicators:
                print(f"ℹ️ 信息: 数据中存在未配置的指标: {extra_indicators}")
            
            print(f"✅ 数据加载成功!")
            print(f"   - 样本数量: {len(self.data)}")
            print(f"   - 指标数量: {len(self.indicator_cols)}")
            print(f"   - 指标列表: {self.indicator_cols}")
            
            return True
            
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            return False
    
    def calculate_indicator_frequency(self):
        """计算子指标存在频率"""
        print("\n📊 计算子指标存在频率...")
        
        frequencies = {}
        for indicator in self.indicator_cols:
            if indicator in self.data.columns:
                # 计算存在频率（值为1的比例）
                frequency = self.data[indicator].sum() / len(self.data)
                frequencies[indicator] = frequency
                print(f"   {indicator}: {frequency:.4f} ({self.data[indicator].sum()}/{len(self.data)})")
        
        self.results['frequencies'] = frequencies
        return frequencies
    
    def calculate_entropy(self):
        """计算信息熵"""
        print("\n🧮 计算信息熵...")
        
        entropies = {}
        for indicator in self.indicator_cols:
            if indicator in self.data.columns:
                # 计算该指标的概率分布
                p1 = self.results['frequencies'][indicator]  # 存在的概率
                p0 = 1 - p1  # 不存在的概率
                
                # 避免log(0)的情况
                if p1 == 0 or p0 == 0:
                    entropy = 0  # 完全确定的情况，熵为0
                else:
                    entropy = -(p1 * np.log2(p1) + p0 * np.log2(p0))
                
                entropies[indicator] = entropy
                print(f"   {indicator}: {entropy:.4f}")
        
        self.results['entropies'] = entropies
        return entropies
    
    def calculate_weights(self):
        """计算权重"""
        print("\n⚖️ 计算权重...")
        
        # 计算信息效用值（1 - 熵值）
        utility_values = {}
        for indicator in self.indicator_cols:
            if indicator in self.results['entropies']:
                utility_values[indicator] = 1 - self.results['entropies'][indicator]
        
        # 按一级维度分组计算权重
        weights = {}
        for primary_dim, indicators in self.indicator_hierarchy.items():
            print(f"\n   {primary_dim} 维度:")
            
            # 提取该维度的指标
            dim_indicators = [ind for ind in indicators if ind in utility_values]
            
            if not dim_indicators:
                print(f"      警告: {primary_dim} 维度没有有效指标")
                continue
            
            # 计算该维度内的权重
            dim_utility_sum = sum([utility_values[ind] for ind in dim_indicators])
            
            if dim_utility_sum == 0:
                # 如果所有指标的效用值都为0，则等权重
                dim_weight = 1.0 / len(dim_indicators)
                for ind in dim_indicators:
                    weights[ind] = dim_weight
                    print(f"      {ind}: {dim_weight:.4f} (等权重)")
            else:
                # 归一化得到权重
                for ind in dim_indicators:
                    weight = utility_values[ind] / dim_utility_sum
                    weights[ind] = weight
                    print(f"      {ind}: {weight:.4f}")
        
        self.results['weights'] = weights
        return weights
    
    def calculate_primary_scores(self):
        """计算一级维度得分"""
        print("\n🎯 计算一级维度得分...")
        
        primary_scores = {}
        
        for idx, row in self.data.iterrows():
            station_name = row['filename']
            primary_scores[station_name] = {}
            
            for primary_dim, indicators in self.indicator_hierarchy.items():
                # 计算该维度的加权得分
                weighted_score = 0
                total_weight = 0
                
                for indicator in indicators:
                    if indicator in self.results['weights'] and indicator in row:
                        indicator_score = row[indicator]  # 0 or 1
                        indicator_weight = self.results['weights'][indicator]
                        
                        weighted_score += indicator_score * indicator_weight
                        total_weight += indicator_weight
                
                # 归一化得分
                if total_weight > 0:
                    normalized_score = weighted_score / total_weight
                else:
                    normalized_score = 0
                
                primary_scores[station_name][primary_dim] = normalized_score
        
        self.results['primary_scores'] = primary_scores
        return primary_scores
    
    def save_entropy_results(self):
        """保存熵权法计算结果"""
        print("\n💾 保存熵权法计算结果...")
        
        # 准备指标详情数据
        indicator_details = []
        for indicator in self.indicator_cols:
            if indicator in self.results['frequencies']:
                row = {
                    'indicator': indicator,
                    'frequency': self.results['frequencies'][indicator],
                    'entropy': self.results['entropies'][indicator],
                    'utility_value': 1 - self.results['entropies'][indicator],
                    'weight': self.results['weights'].get(indicator, 0)
                }
                
                # 查找所属一级维度
                for primary_dim, indicators in self.indicator_hierarchy.items():
                    if indicator in indicators:
                        row['primary_dimension'] = primary_dim
                        break
                else:
                    row['primary_dimension'] = '未分类'
                
                indicator_details.append(row)
        
        # 保存指标详情
        indicator_df = pd.DataFrame(indicator_details)
        indicator_file = self.output_dir / 'entropy_weights_details.csv'
        indicator_df.to_csv(indicator_file, index=False, encoding='utf-8-sig')
        print(f"✅ 指标详情已保存: {indicator_file}")
        
        # 准备车站得分数据
        station_scores = []
        for station_name, scores in self.results['primary_scores'].items():
            row = {'station_name': station_name}
            row.update(scores)
            
            # 计算综合得分（等权重平均）
            row['comprehensive_score'] = np.mean(list(scores.values()))
            station_scores.append(row)
        
        # 保存车站得分
        station_df = pd.DataFrame(station_scores)
        station_file = self.output_dir / 'entropy_station_scores.csv'
        station_df.to_csv(station_file, index=False, encoding='utf-8-sig')
        print(f"✅ 车站得分已保存: {station_file}")
        
        return indicator_file, station_file

class MonteCarloOptimizer:
    """蒙特卡洛模拟优化器"""
    
    def __init__(self, data, indicator_hierarchy, primary_scores, secondary_weights, 
                 iterations=10000, output_dir=None):
        self.data = data
        self.indicator_hierarchy = indicator_hierarchy
        self.primary_scores = primary_scores
        self.secondary_weights = secondary_weights
        self.iterations = iterations
        self.output_dir = Path(output_dir) if output_dir else Path('.')
        self.results = {}
        
    def generate_random_primary_weights(self):
        """生成符合约束的随机一级权重"""
        primary_dims = list(self.indicator_hierarchy.keys())
        
        # 如果有约束，在约束范围内生成
        if PRIMARY_WEIGHT_CONSTRAINTS:
            weights = []
            for dim in primary_dims:
                if dim in PRIMARY_WEIGHT_CONSTRAINTS:
                    min_w, max_w = PRIMARY_WEIGHT_CONSTRAINTS[dim]
                    weight = np.random.uniform(min_w, max_w)
                else:
                    weight = np.random.uniform(0.1, 0.8)
                weights.append(weight)
            
            # 归一化
            weights = np.array(weights)
            weights = weights / weights.sum()
        else:
            # 无约束情况，使用狄利克雷分布生成
            weights = np.random.dirichlet([1] * len(primary_dims))
        
        return dict(zip(primary_dims, weights))
    
    def calculate_comprehensive_score(self, station_name, primary_weights):
        """计算车站综合得分"""
        if station_name not in self.primary_scores:
            return 0
        
        comprehensive_score = 0
        for primary_dim, weight in primary_weights.items():
            if primary_dim in self.primary_scores[station_name]:
                dim_score = self.primary_scores[station_name][primary_dim]
                comprehensive_score += dim_score * weight
        
        return comprehensive_score
    
    def run_monte_carlo_simulation(self):
        """运行蒙特卡洛模拟"""
        print(f"\n🎲 开始蒙特卡洛模拟 ({self.iterations:,} 次迭代)...")
        
        station_names = list(self.primary_scores.keys())
        all_scores = {station: [] for station in station_names}
        all_primary_weights = []
        
        start_time = time.time()
        
        for i in range(self.iterations):
            # 生成随机一级权重
            primary_weights = self.generate_random_primary_weights()
            all_primary_weights.append(primary_weights)
            
            # 计算每个车站的综合得分
            for station in station_names:
                score = self.calculate_comprehensive_score(station, primary_weights)
                all_scores[station].append(score)
            
            # 显示进度
            if (i + 1) % 1000 == 0:
                progress = (i + 1) / self.iterations * 100
                elapsed = time.time() - start_time
                eta = elapsed / (i + 1) * (self.iterations - i - 1)
                print(f"   进度: {progress:.1f}% ({i+1:,}/{self.iterations:,}) - "
                      f"已用时: {elapsed:.1f}s, 预计剩余: {eta:.1f}s")
        
        print(f"✅ 模拟完成! 总用时: {time.time() - start_time:.2f} 秒")
        
        self.results['all_scores'] = all_scores
        self.results['all_primary_weights'] = all_primary_weights
        return all_scores, all_primary_weights
    
    def calculate_stability_metrics(self):
        """计算稳定性指标"""
        print("\n📈 计算稳定性指标...")
        
        all_scores = self.results['all_scores']
        station_stds = {}
        
        for station, scores in all_scores.items():
            station_stds[station] = np.std(scores)
        
        # 计算平均标准差
        avg_std = np.mean(list(station_stds.values()))
        
        print(f"   各车站得分标准差:")
        for station, std in sorted(station_stds.items(), key=lambda x: x[1]):
            print(f"     {station}: {std:.4f}")
        
        print(f"\n   平均标准差: {avg_std:.4f}")
        
        self.results['station_stds'] = station_stds
        self.results['avg_std'] = avg_std
        
        return station_stds, avg_std
    
    def find_optimal_weights(self):
        """找到最优权重组合"""
        print("\n🎯 寻找最优权重组合...")
        
        all_scores = self.results['all_scores']
        all_primary_weights = self.results['all_primary_weights']
        
        # 计算每次迭代的平均标准差
        iteration_stds = []
        for i in range(self.iterations):
            scores_at_i = [all_scores[station][i] for station in all_scores.keys()]
            iteration_stds.append(np.std(scores_at_i))
        
        # 找到标准差最小的迭代
        optimal_idx = np.argmin(iteration_stds)
        optimal_primary_weights = all_primary_weights[optimal_idx]
        optimal_std = iteration_stds[optimal_idx]
        
        print(f"   最优权重组合 (第 {optimal_idx + 1} 次迭代):")
        for dim, weight in optimal_primary_weights.items():
            print(f"     {dim}: {weight:.4f}")
        
        print(f"   该组合下的标准差: {optimal_std:.4f}")
        
        self.results['optimal_weights'] = optimal_primary_weights
        self.results['optimal_std'] = optimal_std
        self.results['optimal_idx'] = optimal_idx
        
        return optimal_primary_weights, optimal_std
    
    def calculate_final_scores(self):
        """使用最优权重计算最终得分"""
        print("\n🏆 计算最终得分...")
        
        optimal_weights = self.results['optimal_weights']
        final_scores = {}
        
        for station in self.primary_scores.keys():
            score = self.calculate_comprehensive_score(station, optimal_weights)
            final_scores[station] = score
        
        # 排序
        sorted_scores = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        
        print("   最终排名:")
        for i, (station, score) in enumerate(sorted_scores[:10]):  # 显示前10名
            print(f"     {i+1:2d}. {station}: {score:.4f}")
        
        self.results['final_scores'] = final_scores
        self.results['ranking'] = sorted_scores
        
        return final_scores, sorted_scores
    
    def save_monte_carlo_results(self):
        """保存蒙特卡洛模拟结果"""
        print("\n💾 保存蒙特卡洛模拟结果...")
        
        # 1. 保存最优权重
        optimal_weights_data = []
        for dim, weight in self.results['optimal_weights'].items():
            optimal_weights_data.append({
                'primary_dimension': dim,
                'optimal_weight': weight
            })
        
        optimal_df = pd.DataFrame(optimal_weights_data)
        optimal_file = self.output_dir / 'optimal_primary_weights.csv'
        optimal_df.to_csv(optimal_file, index=False, encoding='utf-8-sig')
        print(f"✅ 最优权重已保存: {optimal_file}")
        
        # 2. 保存最终得分和排名
        final_scores_data = []
        for i, (station, score) in enumerate(self.results['ranking']):
            # 获取原始数据
            station_info = self.data[self.data['filename'] == station].iloc[0] if len(
                self.data[self.data['filename'] == station]) > 0 else {}
            
            row = {
                'rank': i + 1,
                'station_name': station,
                'comprehensive_score': score,
                'score_std': self.results['station_stds'].get(station, 0)
            }
            
            # 添加地理信息
            if 'longitude' in station_info:
                row['longitude'] = station_info['longitude']
            if 'latitude' in station_info:
                row['latitude'] = station_info['latitude']
            
            # 添加一级维度得分
            if station in self.primary_scores:
                for dim, dim_score in self.primary_scores[station].items():
                    row[f'{dim}_score'] = dim_score
            
            final_scores_data.append(row)
        
        final_df = pd.DataFrame(final_scores_data)
        final_file = self.output_dir / 'final_station_ranking.csv'
        final_df.to_csv(final_file, index=False, encoding='utf-8-sig')
        print(f"✅ 最终排名已保存: {final_file}")
        
        # 3. 保存模拟统计信息
        simulation_stats = {
            'total_iterations': self.iterations,
            'avg_std_across_stations': self.results['avg_std'],
            'optimal_iteration': self.results['optimal_idx'] + 1,
            'optimal_std': self.results['optimal_std'],
            'total_stations': len(self.primary_scores),
            'simulation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        stats_df = pd.DataFrame([simulation_stats])
        stats_file = self.output_dir / 'monte_carlo_stats.csv'
        stats_df.to_csv(stats_file, index=False, encoding='utf-8-sig')
        print(f"✅ 模拟统计已保存: {stats_file}")
        
        return optimal_file, final_file, stats_file

def main():
    """主函数"""
    print("🚀 熵权法+蒙特卡洛模拟优化权重计算程序")
    print("="*60)
    
    # 检查输入文件
    if not Path(INPUT_CSV).exists():
        print(f"❌ 错误: 输入文件不存在: {INPUT_CSV}")
        print("请确保CSV文件路径正确")
        return
    
    # 创建输出目录
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 第一阶段：熵权法计算
        print("\n" + "="*60)
        print("第一阶段：熵权法计算子指标权重")
        print("="*60)
        
        entropy_calc = EntropyWeightCalculator(INPUT_CSV, OUTPUT_DIR, INDICATOR_HIERARCHY)
        
        # 加载数据
        if not entropy_calc.load_data():
            return
        
        # 计算频率、熵值、权重
        entropy_calc.calculate_indicator_frequency()
        entropy_calc.calculate_entropy()
        entropy_calc.calculate_weights()
        entropy_calc.calculate_primary_scores()
        
        # 保存熵权法结果
        entropy_calc.save_entropy_results()
        
        # 第二阶段：蒙特卡洛模拟优化
        print("\n" + "="*60)
        print("第二阶段：蒙特卡洛模拟优化一级权重")
        print("="*60)
        
        # 设置随机种子
        np.random.seed(RANDOM_SEED)
        
        mc_optimizer = MonteCarloOptimizer(
            entropy_calc.data,
            INDICATOR_HIERARCHY,
            entropy_calc.results['primary_scores'],
            entropy_calc.results['weights'],
            MONTE_CARLO_ITERATIONS,
            OUTPUT_DIR
        )
        
        # 运行蒙特卡洛模拟
        mc_optimizer.run_monte_carlo_simulation()
        mc_optimizer.calculate_stability_metrics()
        mc_optimizer.find_optimal_weights()
        mc_optimizer.calculate_final_scores()
        
        # 保存蒙特卡洛结果
        mc_optimizer.save_monte_carlo_results()
        
        # 总结
        print("\n" + "="*60)
        print("🎉 计算完成！")
        print("="*60)
        print(f"输出文件:")
        print(f"  1. entropy_weights_details.csv - 熵权法指标详情")
        print(f"  2. entropy_station_scores.csv - 熵权法车站得分")
        print(f"  3. optimal_primary_weights.csv - 最优一级权重")
        print(f"  4. final_station_ranking.csv - 最终车站排名")
        print(f"  5. monte_carlo_stats.csv - 蒙特卡洛模拟统计")
        print(f"\n所有文件已保存到: {OUTPUT_DIR}")
        
    except Exception as e:
        print(f"\n❌ 程序执行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
