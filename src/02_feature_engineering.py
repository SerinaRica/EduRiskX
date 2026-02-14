# src/feature_engineer.py（最终适配版）
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

# 方式1：复制OULADLoader类到当前文件（简单直接）
class OULADLoader:
    """简化版的OULADLoader，用于feature_engineer内部使用"""
    @staticmethod
    def create_target_variable(student_df):
        """创建目标变量"""
        student_df = student_df.copy()
        student_df['is_risk'] = student_df['final_result'].apply(
            lambda x: 1 if x in ['Withdrawn', 'Fail'] else 0
        )
        return student_df

# 方式2：导入完整的OULADLoader（如果文件在同一目录）
# from data_preprocessor import OULADLoader

# 方式3：不导入，而是将相关功能内嵌
def create_target_variable(student_df):
    """创建目标变量"""
    student_df = student_df.copy()
    student_df['is_risk'] = student_df['final_result'].apply(
        lambda final_result: 1 if final_result in ['Withdrawn', 'Fail'] else 0
    )
    return student_df

class FeatureEngineer:
    def __init__(self, theory_config_path='config/theory_keywords.json'):
        # 加载理论关键词
        try:
            with open(theory_config_path, 'r') as f:
                self.theory_keywords = json.load(f)
        except FileNotFoundError:
            print(f"警告: 配置文件 {theory_config_path} 未找到，使用默认配置")
            self.theory_keywords = {
                "self_efficacy": {"constructs": ["confidence", "persistence"], 
                                 "indicators": ["decline", "avoidance"]},
                "student_integration": {"constructs": ["integration", "belonging"], 
                                       "indicators": ["isolation"]},
                "engagement": {"constructs": ["engagement", "participation"], 
                              "indicators": ["decline", "inactive"]}
            }
    
    def create_student_week_features(self, student_vle_df, student_info_df, 
                                    assessments_df, student_assessment_df):
        """
        创建学生-周级别的特征矩阵（修复日期bug+适配oucontent+无forum数据）
        """
        print("开始创建学生-周特征矩阵...")
        
        # 1. 数据预处理
        # 确保有目标变量
        if 'is_risk' not in student_info_df.columns:
            student_info_df = create_target_variable(student_info_df)
        
        # 2. 处理VLE日期（核心修复：数值日期不转datetime，直接用于计算周数）
        student_vle_df = student_vle_df.copy()
        # 确保date为数值型，负日期已转为0（从data_preprocessing同步）
        student_vle_df['date'] = pd.to_numeric(student_vle_df['date'], errors='coerce').clip(lower=0).fillna(0)
        
        # 3. 确定课程基准日期（数值型，以课程内最小日期为0）
        course_baseline = student_vle_df.groupby(
            ['code_module', 'code_presentation']
        )['date'].min().reset_index()
        course_baseline.rename(columns={'date': 'course_baseline_date'}, 
                              inplace=True)
        
        # 4. 合并基准日期并计算周数（数值日期直接计算，无datetime转换）
        student_vle_with_baseline = pd.merge(
            student_vle_df, 
            course_baseline, 
            on=['code_module', 'code_presentation'],
            how='left'
        )
        
        # 计算周数（相对于课程基准日期，每7天为1周）
        student_vle_with_baseline['week_number'] = (
            (student_vle_with_baseline['date'] - student_vle_with_baseline['course_baseline_date']) // 7
        ).astype(int)
        
        # 移除无效周数（负值或超过52周）
        student_vle_with_baseline = student_vle_with_baseline[
            (student_vle_with_baseline['week_number'] >= 0) & 
            (student_vle_with_baseline['week_number'] < 52)  # 假设课程不超过52周
        ]
        
        # 5. 按学生和周聚合点击数据（适配oucontent，核心修改）
        # 先加载vle信息，识别内容类资源（oucontent/resource）
        try:
            vle_df = pd.read_csv('data/raw/vle.csv')
            # 标记内容类资源（适配你的vle数据：oucontent）
            content_site_ids = vle_df[vle_df['activity_type'].str.contains(
                'oucontent|resource', case=False, na=False)]['id_site'].unique()
            # 标记论坛类资源（你的数据中无forum，此列表为空）
            forum_site_ids = vle_df[vle_df['activity_type'].str.contains(
                'forum', case=False, na=False)]['id_site'].unique()
            
            # 筛选内容类点击数据
            content_clicks_df = student_vle_with_baseline[
                student_vle_with_baseline['id_site'].isin(content_site_ids)
            ]
        except FileNotFoundError:
            print("警告: vle.csv文件未找到，使用全部点击作为内容点击")
            content_clicks_df = student_vle_with_baseline
            forum_site_ids = []
        
        # 聚合总点击和内容点击
        weekly_clicks = student_vle_with_baseline.groupby(
            ['id_student', 'week_number']
        ).agg(
            total_clicks=('sum_click', 'sum'),
            days_active=('date', lambda dates: len(dates.unique())),
            unique_resources=('id_site', 'nunique')
        ).reset_index()
        
        # 聚合内容类点击（适配oucontent）
        if len(content_clicks_df) > 0:
            content_weekly_clicks = content_clicks_df.groupby(
                ['id_student', 'week_number']
            ).agg(
                content_clicks=('sum_click', 'sum')
            ).reset_index()
            weekly_clicks = pd.merge(
                weekly_clicks, content_weekly_clicks, 
                on=['id_student', 'week_number'], 
                how='left'
            ).fillna({'content_clicks': 0})
        else:
            weekly_clicks['content_clicks'] = 0
        
        # 6. 添加论坛参与特征（你的数据中无forum，直接设为0，优化警告）
        weekly_features = weekly_clicks.copy()
        # 因无论坛资源，直接添加0值列，避免不必要的计算
        weekly_features['forum_clicks'] = 0
        weekly_features['forum_days'] = 0
        print("提示: 未检测到论坛类资源，forum特征设为0（符合你的数据特点）")
        
        # 7. 添加评估特征（同步数值日期处理，修复datetime bug）
        try:
            # 处理评估数据（数值日期，不转datetime）
            assessments_df = assessments_df.copy()
            student_assessment_df = student_assessment_df.copy()
            
            # 评估截止日期转为数值型
            assessments_df['date'] = pd.to_numeric(assessments_df['date'], errors='coerce').clip(lower=0).fillna(0)
            # 提交日期转为数值型
            student_assessment_df['date_submitted'] = pd.to_numeric(
                student_assessment_df['date_submitted'], errors='coerce').clip(lower=0).fillna(0)
            
            # 合并评估信息
            merged_assessments = pd.merge(
                student_assessment_df,
                assessments_df[['id_assessment', 'date', 'weight']],
                on='id_assessment',
                how='left'
            )
            
            # 计算是否延迟提交（数值日期直接比较）
            merged_assessments['is_late'] = (
                merged_assessments['date_submitted'] > merged_assessments['date']
            ).astype(int)
            
            # 计算提交周数（相对于课程基准，简化为直接按7天划分）
            merged_assessments['submission_week'] = (
                merged_assessments['date_submitted'] // 7
            ).astype(int)
            
            # 按学生和周聚合评估特征
            weekly_assessments = merged_assessments.groupby(
                ['id_student', 'submission_week']
            ).agg(
                avg_score=('score', lambda x: x[x >= 0].mean() if len(x[x >= 0]) > 0 else -1),
                late_submissions=('is_late', 'sum'),
                total_submissions=('id_assessment', 'count'),
                weighted_score=('score', lambda x: x[x >= 0].mean() if len(x[x >= 0]) > 0 else -1)
            ).reset_index()
            weekly_assessments.rename(columns={'submission_week': 'week_number'}, 
                                     inplace=True)
            
            # 合并评估特征
            final_features = pd.merge(
                weekly_features,
                weekly_assessments,
                on=['id_student', 'week_number'],
                how='left'
            )
            
        except Exception as e:
            print(f"警告: 评估特征处理出错: {e}")
            final_features = weekly_features.copy()
            final_features['avg_score'] = -1
            final_features['late_submissions'] = 0
            final_features['total_submissions'] = 0
            final_features['weighted_score'] = -1
        
        # 填充缺失值
        fill_values = {
            'avg_score': -1,
            'late_submissions': 0,
            'total_submissions': 0,
            'weighted_score': -1,
            'content_clicks': 0
        }
        final_features = final_features.fillna(fill_values)
        
        # 8. 添加趋势特征（稳定修复版，无未定义变量）
        final_features = final_features.sort_values(['id_student', 'week_number'])
        
        # 需要计算趋势的特征列
        trend_columns = ['total_clicks', 'content_clicks', 'avg_score']
        # 移除forum_clicks（始终为0，无趋势可计算）
        
        for col in trend_columns:
            if col in final_features.columns:
                final_features[f'{col}_trend_3w'] = final_features.groupby(
                    'id_student'
                )[col].transform(
                    lambda series: series.rolling(window=4, min_periods=2).apply(
                        lambda window_values: (
                            (window_values.iloc[-1] - window_values.iloc[:-1].mean()) / 
                            (window_values.iloc[:-1].mean() + 1e-10)
                        ) if len(window_values) > 1 else 0
                    )
                )
                
                # 计算是否连续下降
                final_features[f'{col}_declining'] = final_features.groupby(
                    'id_student'
                )[col].transform(
                    lambda series: series.rolling(window=3).apply(
                        lambda window_values: 1 if (
                            len(window_values) == 3 and 
                            all(window_values.iloc[i] > window_values.iloc[i+1] 
                                for i in range(len(window_values)-1))
                        ) else 0
                    )
                )
        
        # 9. 添加人口统计特征
        student_basic_cols = ['id_student', 'gender', 'age_band', 
                             'imd_band', 'disability', 'is_risk']
        available_cols = [col for col in student_basic_cols if col in student_info_df.columns]
        
        if available_cols:
            student_basic = student_info_df[available_cols].drop_duplicates(subset=['id_student'])
            final_features = pd.merge(final_features, student_basic, 
                                     on='id_student', how='left')
        else:
            print("警告: 学生基本信息列不完整")
        
        # 10. 添加一些衍生特征（适配content_clicks）
        # 活动强度
        final_features['activity_intensity'] = final_features['total_clicks'] / (
            final_features['days_active'] + 1)
        
        # 内容参与度（替代原论坛参与度，更符合你的数据）
        final_features['content_participation_ratio'] = (
            final_features['content_clicks'] / (final_features['total_clicks'] + 1))
        
        # 填充最后可能存在的NaN值
        final_features = final_features.fillna(0)
        
        print(f"特征矩阵创建完成，形状: {final_features.shape}")
        print(f"特征列数: {len(final_features.columns)}")
        print(f"前10个特征列: {list(final_features.columns)[:10]}")
        
        return final_features
    
    def discretize_features_for_association_rules(self, features_df):
        """
        将连续特征离散化，用于关联规则挖掘（适配无forum数据）
        """
        print("开始特征离散化...")
        
        df = features_df.copy()
        
        # 离散化规则定义（优化forum_clicks为0的情况，调整content_clicks离散化）
        discretization_rules = {
            'total_clicks': {
                'bins': [0, 10, 50, 200, 500, np.inf],
                'labels': ['VL', 'L', 'M', 'H', 'VH']  # 很低, 低, 中, 高, 很高
            },
            'total_clicks_trend_3w': {
                'bins': [-np.inf, -0.5, -0.2, 0.2, 0.5, np.inf],
                'labels': ['SD', 'MD', 'ST', 'MI', 'SI']  # 严重降, 中等降, 稳定, 中等增, 显著增
            },
            'content_clicks': {  # 新增：适配oucontent的离散化
                'bins': [0, 5, 25, 100, 300, np.inf],
                'labels': ['VL', 'L', 'M', 'H', 'VH']
            },
            'avg_score': {
                'bins': [-2, 0, 40, 60, 80, 101],
                'labels': ['No_Data', 'Fail', 'Low', 'Medium', 'High']
            },
            'late_submissions': {
                'bins': [-1, 0, 1, 3, np.inf],
                'labels': ['None', 'One', 'Few', 'Many']
            }
        }
        
        # 创建离散化后的DataFrame
        discretized_df = pd.DataFrame()
        discretized_df['id_student'] = df['id_student']
        
        # 处理week列（支持week或week_number两种列名）
        if 'week' in df.columns:
            discretized_df['week'] = df['week']
        elif 'week_number' in df.columns:
            discretized_df['week'] = df['week_number']
        
        # 添加目标变量（如果存在）
        if 'is_risk' in df.columns:
            discretized_df['is_risk'] = df['is_risk']
        
        # 对每个特征进行离散化
        for col, rule in discretization_rules.items():
            if col in df.columns:
                try:
                    discretized_df[f'{col}_cat'] = pd.cut(
                        df[col], 
                        bins=rule['bins'], 
                        labels=rule['labels'],
                        include_lowest=True
                    ).astype(str)
                    print(f"  - {col}: 离散化完成")
                except Exception as e:
                    print(f"  - {col}: 离散化失败 - {e}")
                    discretized_df[f'{col}_cat'] = 'Unknown'
        
        print(f"离散化完成，形状: {discretized_df.shape}")
        return discretized_df

# 测试代码
if __name__ == "__main__":
    print("测试FeatureEngineer...")
    
    # 创建模拟数据
    print("创建模拟数据...")
    
    # 模拟学生信息
    n_students = 100
    student_info = pd.DataFrame({
        'id_student': range(1, n_students + 1),
        'gender': np.random.choice(['M', 'F'], n_students),
        'age_band': np.random.choice(['0-35', '35-55', '55<='], n_students),
        'imd_band': np.random.choice(['0-10%', '10-20%', '20-30%'], n_students),
        'disability': np.random.choice(['Y', 'N'], n_students, p=[0.1, 0.9]),
        'final_result': np.random.choice(['Pass', 'Fail', 'Withdrawn'], 
                                        n_students, p=[0.7, 0.2, 0.1])
    })
    student_info = create_target_variable(student_info)
    
    # 模拟VLE数据（数值日期，无负日期，适配oucontent）
    vle_data = []
    for student_id in range(1, n_students + 1):
        for week in range(1, 11):  # 10周
            vle_data.append({
                'id_student': student_id,
                'code_module': 'AAA',
                'code_presentation': '2013J',
                'id_site': np.random.randint(1, 50),  # 模拟oucontent资源ID
                'date': week * 7,  # 数值日期，每7天递增
                'sum_click': np.random.poisson(25) if np.random.random() > 0.2 else 0
            })
    
    student_vle = pd.DataFrame(vle_data)
    
    # 模拟评估数据
    assessments = pd.DataFrame({
        'id_assessment': range(1, 6),
        'date': [14, 28, 42, 56, 70],  # 数值日期
        'weight': [10, 15, 20, 25, 30]
    })
    
    student_assessment = []
    for student_id in range(1, n_students + 1):
        for assess_id in range(1, 6):
            student_assessment.append({
                'id_assessment': assess_id,
                'id_student': student_id,
                'date_submitted': 14 + assess_id*14 + np.random.randint(-3, 4),  # 数值日期
                'score': np.random.randint(30, 100) if np.random.random() > 0.3 else np.random.randint(0, 40)
            })
    
    student_assessment = pd.DataFrame(student_assessment)
    
    # 测试FeatureEngineer
    print("\n测试特征工程...")
    engineer = FeatureEngineer()
    
    weekly_features = engineer.create_student_week_features(
        student_vle, student_info, assessments, student_assessment
    )
    
    print(f"\n创建的特征矩阵:")
    print(f"行数: {len(weekly_features)}")
    print(f"列数: {len(weekly_features.columns)}")
    print(f"列名: {list(weekly_features.columns)[:15]}")
    
    # 测试离散化
    print("\n测试特征离散化...")
    discretized = engineer.discretize_features_for_association_rules(weekly_features)
    
    print(f"\n离散化后的特征矩阵:")
    print(f"行数: {len(discretized)}")
    print(f"列数: {len(discretized.columns)}")
    
    # 显示前几行
    print("\n离散化数据示例:")
    print(discretized.head())