# src/01_data_preprocessing.py
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class OULADLoader:
    def __init__(self, data_path='data/raw/'):
        self.data_path = Path(data_path)
        
    def load_all_tables(self):
        """加载所有OULAD表（同步清洗负日期+缺失值）"""
        tables = {}
        
        # 1. 学生基本信息（标签来源）
        tables['student_info'] = pd.read_csv(self.data_path / 'studentInfo.csv')
        # 简单清洗学生信息缺失值
        tables['student_info'] = self._clean_student_info(tables['student_info'])
        
        # 2. 学生虚拟学习环境交互（核心行为数据）
        # 注意：studentVle.csv很大（约1.2GB），分块读取+处理负日期
        chunks = []
        for chunk in pd.read_csv(self.data_path / 'studentVle.csv', 
                                chunksize=1000000):
            # 处理负日期（核心修改：将负日期转为0，代表课程开始前）
            chunk = self._clean_student_vle_chunk(chunk)
            chunks.append(chunk)
        tables['student_vle'] = pd.concat(chunks, ignore_index=True)
        
        # 3. 评估相关数据
        tables['student_assessment'] = pd.read_csv(
            self.data_path / 'studentAssessment.csv')
        tables['assessments'] = pd.read_csv(self.data_path / 'assessments.csv')
        # 清洗评估数据缺失值
        tables['student_assessment'] = self._clean_student_assessment(tables['student_assessment'])
        
        # 4. 注册信息
        tables['student_registration'] = pd.read_csv(
            self.data_path / 'studentRegistration.csv')
        
        # 5. 资源和课程信息
        tables['vle'] = pd.read_csv(self.data_path / 'vle.csv')
        tables['courses'] = pd.read_csv(self.data_path / 'courses.csv')
        # 适配vle的activity_type（核心修改：标记oucontent为内容类资源）
        tables['vle']['is_content'] = tables['vle']['activity_type'].str.contains('oucontent|resource', case=False, na=False)
        tables['vle']['is_forum'] = tables['vle']['activity_type'].str.contains('forum', case=False, na=False)
        
        return tables
    
    def _clean_student_vle_chunk(self, chunk):
        """清洗单个studentVle分块（处理负日期+缺失值）"""
        chunk = chunk.copy()
        # 1. 负日期转为0（课程开始前的互动统一记为第0周）
        if 'date' in chunk.columns:
            chunk['date'] = chunk['date'].clip(lower=0)
        # 2. 填充缺失的sum_click为0
        if 'sum_click' in chunk.columns:
            chunk['sum_click'] = chunk['sum_click'].fillna(0).astype(int)
        # 3. 移除无效的id_student（非数值）
        if 'id_student' in chunk.columns:
            chunk = chunk[pd.to_numeric(chunk['id_student'], errors='coerce').notna()]
        return chunk
    
    def _clean_student_info(self, student_df):
        """清洗学生信息表"""
        student_df = student_df.copy()
        # 填充缺失的人口统计信息为'Unknown'
        for col in ['gender', 'age_band', 'imd_band', 'disability']:
            if col in student_df.columns:
                student_df[col] = student_df[col].fillna('Unknown')
        return student_df
    
    def _clean_student_assessment(self, assessment_df):
        """清洗学生评估表"""
        assessment_df = assessment_df.copy()
        # 填充缺失的score为-1（标记无成绩）
        if 'score' in assessment_df.columns:
            assessment_df['score'] = assessment_df['score'].fillna(-1)
        # 填充缺失的date_submitted为0
        if 'date_submitted' in assessment_df.columns:
            assessment_df['date_submitted'] = assessment_df['date_submitted'].clip(lower=0).fillna(0)
        return assessment_df
    
    def create_target_variable(self, student_df):
        """
        创建目标变量：是否退学/失败
        final_result: Withdrawn, Fail -> 1 (风险), Pass, Distinction -> 0
        """
        student_df = student_df.copy()
        student_df['is_risk'] = student_df['final_result'].apply(
            lambda x: 1 if x in ['Withdrawn', 'Fail'] else 0
        )
        return student_df

# 使用示例
if __name__ == "__main__":
    loader = OULADLoader()
    data = loader.load_all_tables()
    print("加载的表:", list(data.keys()))
    print(f"学生数量: {len(data['student_info'])}")
    print(f"VLE记录数: {len(data['student_vle']):,}")
    print(f"VLE内容类资源数: {data['vle']['is_content'].sum()}")
    print(f"VLE论坛类资源数: {data['vle']['is_forum'].sum()}")  # 应为0，符合你的数据特点