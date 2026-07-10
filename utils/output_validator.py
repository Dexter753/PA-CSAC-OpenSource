import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


class OutputValidator:
    """
    输出验证器 - 确保最终成果符合预设标准
    """
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.validation_results = {}
    
    def validate_result_directory(self, result_dir: str) -> bool:
        """验证结果目录结构"""
        if not os.path.exists(result_dir):
            self.errors.append(f"结果目录不存在: {result_dir}")
            return False
        
        required_subdirs = ['traces', 'models']
        for subdir in required_subdirs:
            subdir_path = os.path.join(result_dir, subdir)
            if not os.path.exists(subdir_path):
                self.warnings.append(f"缺少子目录: {subdir}")
        
        return True
    
    def validate_summary_file(self, summary_path: str) -> Tuple[bool, Optional[pd.DataFrame]]:
        """验证摘要文件"""
        if not os.path.exists(summary_path):
            self.errors.append(f"摘要文件不存在: {summary_path}")
            return False, None
        
        try:
            df = pd.read_csv(summary_path)
        except Exception as e:
            self.errors.append(f"读取摘要文件失败: {str(e)}")
            return False, None
        
        required_cols = ['fuel_l_per_100km', 'violation_rate', 'gap_rmse']
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            self.errors.append(f"摘要文件缺少必需列: {', '.join(missing_cols)}")
            return False, df
        advisory_cols = ['paper_valid', 'valid_episode_ratio']
        missing_advisory = [c for c in advisory_cols if c not in df.columns]
        if missing_advisory:
            self.warnings.append(f"摘要文件缺少论文有效性列: {', '.join(missing_advisory)}")
        
        if df.empty:
            self.errors.append("摘要文件为空")
            return False, df
        
        return True, df
    
    def validate_fuel_consumption(self, df: pd.DataFrame) -> bool:
        """验证油耗数据的合理性"""
        success = True
        if 'fuel_l_per_100km' in df.columns:
            fc = df['fuel_l_per_100km']
            if fc.min() < 0:
                self.errors.append("油耗数据存在负值")
                success = False
            if fc.max() > 20:
                self.warnings.append(f"油耗数据超过合理范围: {fc.max():.2f} L/100km")
        
        return success
    
    def validate_violation_rate(self, df: pd.DataFrame) -> bool:
        """验证违规率的合理性"""
        success = True
        if 'violation_rate' in df.columns:
            vr = df['violation_rate']
            if (vr < 0).any():
                self.errors.append("违规率存在负值")
                success = False
            if (vr > 1).any():
                self.errors.append("违规率超过100%")
                success = False
        
        return success
    
    def validate_model_files(self, model_dir: str) -> bool:
        """验证模型文件"""
        if not os.path.exists(model_dir):
            self.warnings.append(f"模型目录不存在: {model_dir}")
            return True
        
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.pt') or f.endswith('.pth')]
        if not model_files:
            self.warnings.append(f"模型目录为空: {model_dir}")
        
        return True
    
    def validate_trace_files(self, trace_dir: str) -> bool:
        """验证轨迹文件"""
        if not os.path.exists(trace_dir):
            self.warnings.append(f"轨迹目录不存在: {trace_dir}")
            return True
        
        trace_files = [f for f in os.listdir(trace_dir) if f.endswith('.csv')]
        if not trace_files:
            self.warnings.append(f"轨迹目录为空: {trace_dir}")
        
        for trace_file in trace_files[:5]:
            trace_path = os.path.join(trace_dir, trace_file)
            try:
                df = pd.read_csv(trace_path)
                if len(df) < 10:
                    self.warnings.append(f"轨迹文件数据过少: {trace_file} ({len(df)}行)")
            except Exception as e:
                self.warnings.append(f"读取轨迹文件失败: {trace_file}")
        
        return True
    
    def validate_statistical_significance(self, df: pd.DataFrame) -> bool:
        """验证统计显著性"""
        if 'seed' in df.columns and len(df['seed'].unique()) < 3:
            self.warnings.append("种子数少于3个，统计显著性可能不足")
        if 'paper_valid' in df.columns and not df['paper_valid'].astype(bool).all():
            invalid_count = int((~df['paper_valid'].astype(bool)).sum())
            self.warnings.append(f"存在未通过论文有效性判定的结果: {invalid_count} 条")
        if 'valid_episode_ratio' in df.columns:
            low_valid = (df['valid_episode_ratio'] < 0.70).sum()
            if low_valid > 0:
                self.warnings.append(f"存在有效场景比例低于0.70的结果: {int(low_valid)} 条")
        
        return True
    
    def validate(self, result_dir: str) -> bool:
        """完整验证流程"""
        self.errors = []
        self.warnings = []
        
        if not self.validate_result_directory(result_dir):
            return False
        
        summary_path = os.path.join(result_dir, 'benchmark_summary.csv')
        success, df = self.validate_summary_file(summary_path)
        
        if df is not None:
            self.validate_fuel_consumption(df)
            self.validate_violation_rate(df)
            self.validate_statistical_significance(df)
        
        model_dir = os.path.join(result_dir, 'models')
        self.validate_model_files(model_dir)
        
        trace_dir = os.path.join(result_dir, 'traces')
        self.validate_trace_files(trace_dir)
        
        self.validation_results['result_dir'] = result_dir
        
        return len(self.errors) == 0
    
    def report(self) -> str:
        """生成验证报告"""
        lines = ["=" * 60, "输出验证报告", "=" * 60]
        
        lines.append(f"\n[验证目录]")
        lines.append(f"  {self.validation_results.get('result_dir', 'N/A')}")
        
        if self.warnings:
            lines.append(f"\n[警告]")
            for i, warn in enumerate(self.warnings, 1):
                lines.append(f"  {i}. {warn}")
        
        if self.errors:
            lines.append(f"\n[错误]")
            for i, error in enumerate(self.errors, 1):
                lines.append(f"  {i}. {error}")
        
        status = "通过" if not self.errors else "失败"
        lines.append(f"\n[验证状态] {status}")
        lines.append("=" * 60)
        
        return "\n".join(lines)


class ResultReporter:
    """
    结果报告生成器
    """
    
    @staticmethod
    def generate_comparison_report(results_df: pd.DataFrame, output_path: str = None) -> str:
        """生成对比报告"""
        lines = ["=" * 70, "实验结果对比报告", "=" * 70]
        fuel_col = 'fuel_consumption' if 'fuel_consumption' in results_df.columns else 'fuel_l_per_100km'
        viol_col = 'gap_violation_rate' if 'gap_violation_rate' in results_df.columns else 'violation_rate'
        gap_col = 'avg_gap' if 'avg_gap' in results_df.columns else 'gap_rmse'
        
        lines.append(f"\n[实验概览]")
        lines.append(f"  算法数量: {len(results_df['algorithm'].unique())}")
        lines.append(f"  种子数量: {len(results_df['seed'].unique())}")
        
        lines.append(f"\n[性能指标对比]")
        lines.append("-" * 70)
        lines.append(f"{'算法':<20} {'油耗(L/100km)':<18} {'违规率(%)':<15} {'平均间距(m)':<15}")
        lines.append("-" * 70)
        
        for algo in results_df['algorithm'].unique():
            algo_df = results_df[results_df['algorithm'] == algo]
            fc_mean = algo_df[fuel_col].mean()
            fc_std = algo_df[fuel_col].std()
            vr_mean = algo_df[viol_col].mean() * 100
            gap_mean = algo_df[gap_col].mean()
            
            lines.append(f"{algo:<20} {fc_mean:.2f}±{fc_std:.2f}         {vr_mean:.2f}            {gap_mean:.2f}")
        
        lines.append("-" * 70)
        
        best_fc_algo = results_df.groupby('algorithm')[fuel_col].mean().idxmin()
        best_fc = results_df.groupby('algorithm')[fuel_col].mean().min()
        
        lines.append(f"\n[最佳性能]")
        lines.append(f"  最低油耗: {best_fc_algo} ({best_fc:.2f} L/100km)")
        
        report = "\n".join(lines)
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
        
        return report
