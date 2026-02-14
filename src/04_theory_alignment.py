# src/04_theory_alignment.py
from sentence_transformers import SentenceTransformer
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

class TheoryAligner:
    def __init__(self, theory_config_path='config/theory_keywords.json'):
        # 加载理论关键词配置
        with open(theory_config_path, 'r') as f:
            self.theory_config = json.load(f)
        
        # 加载预训练的句子嵌入模型
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 为每个理论预计算关键词嵌入
        self.theory_embeddings = {}
        for theory_name, theory_info in self.theory_config.items():
            # 合并constructs和indicators
            all_keywords = (theory_info.get('constructs', []) + 
                          theory_info.get('indicators', []))
            if all_keywords:
                self.theory_embeddings[theory_name] = self.model.encode(
                    all_keywords, convert_to_tensor=True)
    
    def calculate_alignment_score(self, rule_text, theory_name):
        """
        计算规则与指定理论的语义对齐度
        """
        if theory_name not in self.theory_embeddings:
            return 0.0
        
        # 编码规则文本
        rule_embedding = self.model.encode(rule_text, 
                                          convert_to_tensor=True)
        
        # 计算与理论所有关键词的最大余弦相似度
        theory_emb_matrix = self.theory_embeddings[theory_name]
        similarities = cosine_similarity(
            rule_embedding.cpu().numpy().reshape(1, -1),
            theory_emb_matrix.cpu().numpy()
        )[0]
        
        # 取平均相似度
        alignment_score = np.mean(similarities)
        
        # 使用sigmoid归一化到(0,1)
        aligned_score = 1 / (1 + np.exp(-10 * (alignment_score - 0.5)))
        
        return aligned_score
    
    def align_all_rules(self, rules_df):
        """
        为所有规则计算理论对齐分数
        """
        aligned_rules = []
        
        for _, rule in rules_df.iterrows():
            rule_text = rule.get('conditions', rule.get('rule', ''))
            
            # 计算与每个理论的相似度
            theory_scores = {}
            total_score = 0.0
            
            for theory_name in ['self_efficacy', 'student_integration', 'engagement']:
                score = self.calculate_alignment_score(rule_text, theory_name)
                theory_scores[f"theory_{theory_name}"] = score
                total_score += score
            
            # 计算综合理论分
            theory_score = total_score / 3 if total_score > 0 else 0
            
            # 创建新的规则记录
            new_rule = rule.copy()
            new_rule.update(theory_scores)
            new_rule['theory_score'] = theory_score
            
            # 计算综合质量分
            confidence = rule.get('confidence', rule.get('accuracy', 0.5))
            support = rule.get('support', 0.01)
            
            quality_score = confidence * theory_score * np.sqrt(support)
            new_rule['quality_score'] = quality_score
            
            aligned_rules.append(new_rule)
        
        return pd.DataFrame(aligned_rules)
    
    def filter_rules_by_quality(self, aligned_rules_df, 
                               min_quality=0.3, 
                               min_support=0.02):
        """
        根据综合质量过滤规则
        """
        filtered = aligned_rules_df[
            (aligned_rules_df['quality_score'] >= min_quality) &
            (aligned_rules_df['support'] >= min_support) &
            (aligned_rules_df['theory_score'] >= 0.3)
        ].copy()
        
        # 按质量分排序
        filtered = filtered.sort_values('quality_score', ascending=False)
        
        # 重置索引
        filtered.reset_index(drop=True, inplace=True)
        filtered['final_rule_id'] = [f"RULE_{i:03d}" for i in range(len(filtered))]
        
        return filtered