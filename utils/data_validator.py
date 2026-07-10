import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


class DataValidator:
    """
    数据验证器 - 确保输入数据的准确性和完整性
    
    该验证器执行以下检查:
    1. 文件存在性检查
    2. CSV结构验证（必需列）
    3. 数值类型验证
    4. NaN/无穷值检测
    5. 数值范围验证
    6. 时间一致性检查
    7. 车辆分组统计验证
    """
    
    REQUIRED_COLUMNS = {
        'Vehicle_ID', 'Timestamp', 'v_ego', 'v_lead', 'd_gap',
        'pred_v_lead_t1', 'std_v_lead_t1', 'ci95_lower_v_lead_t1', 'ci95_upper_v_lead_t1',
        'density', 'flow_speed', 'lead_headway',
    }
    
    COLUMN_RANGES = {
        'v_ego': (0.0, 120.0),
        'v_lead': (0.0, 120.0),
        'd_gap': (0.0, 500.0),
        'std_v_lead_t1': (0.0, 50.0),
        'ci95_lower_v_lead_t1': (-50.0, 120.0),
        'ci95_upper_v_lead_t1': (0.0, 200.0),
        'density': (0.0, 500.0),
        'flow_speed': (0.0, 120.0),
        'lead_headway': (0.0, 200.0),
    }
    
    ALLOWED_NAN_COLUMNS = {
        'true_flow_t2', 'true_v_lead_t2', 'true_flow_t3', 'true_v_lead_t3',
        'true_flow_t4', 'true_v_lead_t4', 'true_flow_t5', 'true_v_lead_t5',
        'true_flow_t6', 'true_v_lead_t6', 'true_flow_t7', 'true_v_lead_t7',
        'true_flow_t8', 'true_v_lead_t8', 'true_flow_t9', 'true_v_lead_t9',
        'true_flow_t10', 'true_v_lead_t10',
    }
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.validation_results = {}
    
    def validate_file_exists(self, file_path: str) -> bool:
        """
        验证文件是否存在
        
        Args:
            file_path: 要验证的文件路径
        
        Returns:
            bool: 文件存在返回True，否则False
        """
        if not os.path.exists(file_path):
            self.errors.append(f"文件不存在: {file_path}")
            return False
        return True
    
    def validate_csv_structure(self, df: pd.DataFrame) -> bool:
        """
        验证CSV结构是否包含所有必需列
        
        Args:
            df: pandas DataFrame
        
        Returns:
            bool: 结构有效返回True，否则False
        """
        missing_cols = self.REQUIRED_COLUMNS - set(df.columns)
        if missing_cols:
            self.errors.append(f"缺失必需列: {', '.join(missing_cols)}")
            return False
        return True
    
    def validate_numeric_columns(self, df: pd.DataFrame) -> bool:
        """
        验证关键列是否为数值类型
        
        Args:
            df: pandas DataFrame
        
        Returns:
            bool: 所有关键列为数值类型返回True，否则False
        """
        success = True
        for col in self.COLUMN_RANGES.keys():
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    self.errors.append(f"列 {col} 不是数值类型")
                    success = False
        return success
    
    def validate_value_ranges(self, df: pd.DataFrame) -> bool:
        """
        验证数值是否在合理范围内
        
        Args:
            df: pandas DataFrame
        
        Returns:
            bool: 所有值在范围内返回True，否则False（警告不影响返回值）
        """
        success = True
        for col, (min_val, max_val) in self.COLUMN_RANGES.items():
            if col in df.columns:
                col_min = df[col].min()
                col_max = df[col].max()
                if col_min < min_val:
                    self.warnings.append(f"列 {col} 存在低于最小值的值: {col_min:.4f}")
                if col_max > max_val:
                    self.warnings.append(f"列 {col} 存在高于最大值的值: {col_max:.4f}")
        return success
    
    def validate_nan_values(self, df: pd.DataFrame) -> bool:
        """
        检查NaN值，区分允许的NaN和问题NaN
        
        Args:
            df: pandas DataFrame
        
        Returns:
            bool: 无问题NaN返回True，否则False
        """
        nan_counts = df.isna().sum()
        nan_cols = nan_counts[nan_counts > 0]
        
        problematic_cols = {k: v for k, v in nan_cols.items() if k not in self.ALLOWED_NAN_COLUMNS}
        
        if problematic_cols:
            self.errors.append(f"存在NaN值的列: {dict(problematic_cols)}")
            return False
        
        allowed_nan_cols = {k: v for k, v in nan_cols.items() if k in self.ALLOWED_NAN_COLUMNS}
        if allowed_nan_cols:
            self.warnings.append(f"允许的NaN值（未来时刻真实值）: {dict(allowed_nan_cols)}")
        
        return True
    
    def validate_infinite_values(self, df: pd.DataFrame) -> bool:
        """
        检查无穷值
        
        Args:
            df: pandas DataFrame
        
        Returns:
            bool: 无无穷值返回True，否则False
        """
        inf_counts = df.isin([np.inf, -np.inf]).sum()
        inf_cols = inf_counts[inf_counts > 0]
        if len(inf_cols) > 0:
            self.errors.append(f"存在无穷值的列: {dict(inf_cols)}")
            return False
        return True
    
    def validate_temporal_consistency(self, df: pd.DataFrame) -> bool:
        """
        验证时间戳的一致性
        
        Args:
            df: pandas DataFrame
        
        Returns:
            bool: 时间戳一致返回True，否则False
        """
        success = True
        if 'Timestamp' in df.columns:
            timestamps = df['Timestamp'].sort_values()
            diffs = timestamps.diff().dropna()
            if (diffs < 0).any():
                self.errors.append("时间戳存在倒序")
                success = False
            if (diffs > 5.0).any():
                self.warnings.append("存在时间间隔大于5秒的数据点")
        return success
    
    def validate_vehicle_groups(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        验证车辆分组的样本分布
        
        Args:
            df: pandas DataFrame
        
        Returns:
            Dict: 包含车辆分组统计信息
        """
        if 'Vehicle_ID' not in df.columns:
            self.errors.append("缺少 Vehicle_ID 列")
            return {}
        
        vehicle_counts = df['Vehicle_ID'].value_counts()
        group_info = {
            'total_vehicles': len(vehicle_counts),
            'min_samples': int(vehicle_counts.min()),
            'max_samples': int(vehicle_counts.max()),
            'mean_samples': float(vehicle_counts.mean()),
        }
        
        if group_info['min_samples'] < 30:
            self.warnings.append(f"存在样本数少于30的车辆组: {group_info['min_samples']}")
        
        return group_info
    
    def validate(self, file_path: str) -> Tuple[bool, Optional[pd.DataFrame]]:
        """
        执行完整的验证流程
        
        Args:
            file_path: 数据文件路径
        
        Returns:
            Tuple[bool, Optional[pd.DataFrame]]: 验证结果和加载的数据（验证失败时为None）
        """
        self.errors = []
        self.warnings = []
        
        if not self.validate_file_exists(file_path):
            return False, None
        
        try:
            df = pd.read_csv(file_path)
        except FileNotFoundError:
            self.errors.append(f"文件不存在: {file_path}")
            return False, None
        except pd.errors.EmptyDataError:
            self.errors.append("文件为空")
            return False, None
        except pd.errors.ParserError:
            self.errors.append("文件格式错误，无法解析CSV")
            return False, None
        except Exception as e:
            self.errors.append(f"读取CSV失败: {str(e)}")
            return False, None
        
        self.validate_csv_structure(df)
        self.validate_numeric_columns(df)
        self.validate_nan_values(df)
        self.validate_infinite_values(df)
        self.validate_value_ranges(df)
        self.validate_temporal_consistency(df)
        self.validation_results['vehicle_groups'] = self.validate_vehicle_groups(df)
        
        self.validation_results.update({
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'columns': list(df.columns),
        })
        
        success = len(self.errors) == 0
        return success, df
    
    def report(self) -> str:
        """
        生成验证报告
        
        Returns:
            str: 格式化的验证报告
        """
        lines = ["=" * 60, "数据验证报告", "=" * 60]
        
        lines.append(f"\n[基本信息]")
        lines.append(f"  总行数: {self.validation_results.get('total_rows', 'N/A')}")
        lines.append(f"  总列数: {self.validation_results.get('total_columns', 'N/A')}")
        
        if 'vehicle_groups' in self.validation_results:
            vg = self.validation_results['vehicle_groups']
            lines.append(f"\n[车辆分组]")
            lines.append(f"  车辆总数: {vg.get('total_vehicles', 'N/A')}")
            lines.append(f"  最小样本数: {vg.get('min_samples', 'N/A')}")
            lines.append(f"  最大样本数: {vg.get('max_samples', 'N/A')}")
            lines.append(f"  平均样本数: {vg.get('mean_samples', 'N/A'):.1f}")
        
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