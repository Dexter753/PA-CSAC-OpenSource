import os
import sys
import time
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from utils.data_validator import DataValidator
    from utils.output_validator import OutputValidator, ResultReporter
except ImportError:
    from .utils.data_validator import DataValidator
    from .utils.output_validator import OutputValidator, ResultReporter


class WorkflowManager:
    """
    工作流管理器 - 实现系统性闭环设计
    确保从数据输入到最终输出的所有环节形成完整、逻辑自洽的工作流程
    """
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else ROOT
        self.logger = self._setup_logger()
        self.data_validator = DataValidator()
        self.output_validator = OutputValidator()
        
        self.workflow_history = []
        self.current_step = 0
        self.errors = []
        self.warnings = []
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger('PA_CSAC_Workflow')
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        log_file = self.project_root / 'workflow.log'
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def _record_step(self, step_name: str, status: str, details: str = ""):
        """记录工作流步骤"""
        self.current_step += 1
        record = {
            'step': self.current_step,
            'name': step_name,
            'status': status,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'details': details,
        }
        self.workflow_history.append(record)
        
        if status == 'SUCCESS':
            self.logger.info(f"步骤 [{self.current_step}] {step_name} - 成功")
        elif status == 'WARNING':
            self.logger.warning(f"步骤 [{self.current_step}] {step_name} - 警告: {details}")
        else:
            self.logger.error(f"步骤 [{self.current_step}] {step_name} - 失败: {details}")
    
    def validate_input_data(self, data_path: str) -> bool:
        """验证输入数据"""
        self.logger.info("开始验证输入数据...")
        
        success, df = self.data_validator.validate(data_path)
        report = self.data_validator.report()
        
        print(report)
        self._save_report(report, 'data_validation_report.txt')
        
        if success:
            self._record_step("输入数据验证", "SUCCESS", f"数据文件: {data_path}")
            return True
        else:
            self.errors.extend(self.data_validator.errors)
            self._record_step("输入数据验证", "FAILED", f"错误数: {len(self.data_validator.errors)}")
            return False
    
    def validate_output_results(self, result_dir: str) -> bool:
        """验证输出结果"""
        self.logger.info("开始验证输出结果...")
        
        success = self.output_validator.validate(result_dir)
        report = self.output_validator.report()
        
        print(report)
        self._save_report(report, 'output_validation_report.txt')
        
        if success:
            self._record_step("输出结果验证", "SUCCESS", f"结果目录: {result_dir}")
            return True
        else:
            self.errors.extend(self.output_validator.errors)
            self.warnings.extend(self.output_validator.warnings)
            self._record_step("输出结果验证", "FAILED", f"错误数: {len(self.output_validator.errors)}")
            return False
    
    def _save_report(self, report: str, filename: str):
        """保存报告文件"""
        report_path = self.project_root / 'reports' / filename
        report_path.parent.mkdir(exist_ok=True)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        self.logger.info(f"报告已保存: {report_path}")
    
    def run_quality_gate(self, result_dir: str) -> Tuple[bool, Dict]:
        """运行质量门检查"""
        self.logger.info("运行质量门检查...")
        
        summary_path = os.path.join(result_dir, 'benchmark_summary.csv')
        if not os.path.exists(summary_path):
            self._record_step("质量门检查", "FAILED", "摘要文件不存在")
            return False, {}
        
        try:
            df = pd.read_csv(summary_path)
        except Exception as e:
            self._record_step("质量门检查", "FAILED", f"读取摘要文件失败: {str(e)}")
            return False, {}
        
        metrics = {}
        all_passed = True
        
        if 'fuel_l_per_100km' in df.columns:
            fc_mean = df['fuel_l_per_100km'].mean()
            metrics['fuel_consumption_mean'] = fc_mean
            if fc_mean > 15.0:
                self.warnings.append(f"平均油耗偏高: {fc_mean:.2f} L/100km")
                all_passed = False
        
        if 'violation_rate' in df.columns:
            vr_mean = df['violation_rate'].mean()
            metrics['violation_rate_mean'] = vr_mean
            if vr_mean > 0.2:
                self.warnings.append(f"违规率偏高: {vr_mean * 100:.2f}%")
                all_passed = False
        
        if 'gap_rmse' in df.columns:
            gap_mean = df['gap_rmse'].mean()
            metrics['avg_gap_rmse'] = gap_mean
            if gap_mean > 50.0:
                self.warnings.append(f"间距RMSE过大: {gap_mean:.2f} m")
                all_passed = False
        if 'valid_episode_ratio' in df.columns:
            valid_ratio_mean = df['valid_episode_ratio'].mean()
            metrics['valid_episode_ratio_mean'] = valid_ratio_mean
            if valid_ratio_mean < 0.70:
                self.warnings.append(f"平均有效场景比例偏低: {valid_ratio_mean:.2%}")
                all_passed = False
        if 'paper_valid' in df.columns and not df['paper_valid'].astype(bool).all():
            invalid_count = int((~df['paper_valid'].astype(bool)).sum())
            metrics['paper_invalid_count'] = invalid_count
            self.warnings.append(f"存在未通过论文有效性判定的结果: {invalid_count} 条")
            all_passed = False
        
        status = "SUCCESS" if all_passed else "WARNING"
        self._record_step("质量门检查", status, str(metrics))
        
        return all_passed, metrics
    
    def generate_final_report(self, output_path: str = None) -> str:
        """生成最终综合报告"""
        lines = ["=" * 70, "PA-CSAC 项目闭环验证报告", "=" * 70]
        lines.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"项目根目录: {self.project_root}")
        lines.append("")
        
        lines.append("[工作流执行记录]")
        lines.append("-" * 70)
        for record in self.workflow_history:
            lines.append(f"步骤 {record['step']}: {record['name']}")
            lines.append(f"  状态: {record['status']}")
            lines.append(f"  时间: {record['timestamp']}")
            if record['details']:
                lines.append(f"  详情: {record['details']}")
            lines.append("")
        
        if self.warnings:
            lines.append("[警告汇总]")
            lines.append("-" * 70)
            for i, warn in enumerate(self.warnings, 1):
                lines.append(f"  {i}. {warn}")
            lines.append("")
        
        if self.errors:
            lines.append("[错误汇总]")
            lines.append("-" * 70)
            for i, error in enumerate(self.errors, 1):
                lines.append(f"  {i}. {error}")
            lines.append("")
        
        overall_status = "失败" if self.errors else ("需复核" if self.warnings else "通过")
        lines.append("[整体状态]")
        lines.append("-" * 70)
        lines.append(f"验证结果: {overall_status}")
        lines.append(f"警告数: {len(self.warnings)}")
        lines.append(f"错误数: {len(self.errors)}")
        lines.append("=" * 70)
        
        report = "\n".join(lines)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            self.logger.info(f"最终报告已保存: {output_path}")
        
        return report
    
    def execute_workflow(self, data_path: str, result_dir: str) -> bool:
        """执行完整工作流"""
        self.logger.info("=" * 70)
        self.logger.info("开始执行 PA-CSAC 工作流")
        self.logger.info("=" * 70)
        
        success = True
        
        if not self.validate_input_data(data_path):
            success = False
        
        if success and not self.validate_output_results(result_dir):
            success = False
        
        if success:
            quality_passed, _ = self.run_quality_gate(result_dir)
            if not quality_passed:
                self.logger.warning("质量门检查未完全通过")
                success = False
        
        final_report = self.generate_final_report(
            os.path.join(self.project_root, 'reports', 'final_verification_report.txt')
        )
        print("\n" + final_report)
        
        return success


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PA-CSAC 工作流管理器")
    parser.add_argument("--data_path", type=str, required=True, help="输入数据路径")
    parser.add_argument("--result_dir", type=str, required=True, help="结果目录路径")
    args = parser.parse_args()
    
    workflow = WorkflowManager()
    success = workflow.execute_workflow(args.data_path, args.result_dir)
    
    sys.exit(0 if success else 1)
