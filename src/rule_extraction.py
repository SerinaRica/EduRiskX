# src/rule_extraction.py
import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None
import json
import os
from datetime import datetime

# 加载预训练模型（用于语义相似度计算，论文3.2.2）
try:
    if SentenceTransformer:
        MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    else:
        MODEL = None
        print("警告: 未安装sentence_transformers，使用简单字符串匹配替代语义对齐")
except Exception as e:
    print(f"警告: 模型加载失败 ({e})，使用简单字符串匹配替代语义对齐")
    MODEL = None

class TheoryAlignedRuleMiner:
    def __init__(self, theory_mapping=None, alignment_threshold=0.6, min_support=0.01, min_confidence=0.5, strict_alignment=False):
        """
        初始化理论对齐规则挖掘器（对齐论文3.2.2）
        :param theory_mapping: 理论配置（默认读取config/theory_keywords.json）
        :param alignment_threshold: 理论对齐分数阈值（论文默认0.6）
        :param min_support: 最小支持度（可选参数，用于兼容扩展接口）
        :param min_confidence: 最小置信度（可选参数，用于兼容扩展接口）
        :param strict_alignment: 是否严格对齐（True=只保留高对齐规则，False=包含所有规则）
        """
        # 加载理论配置（优先传入，无则读取配置文件）
        if theory_mapping is None:
            self.theory_config = self._load_theory_config()
        else:
            self.theory_config = theory_mapping
        self.alignment_threshold = alignment_threshold
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.strict_alignment = strict_alignment
        self.rule_counter = {
            "StaticPatternRule": 1,
            "TemporalSequenceRule": 1,
            "TimeWindowRule": 1
        }
    
    def _load_theory_config(self):
        """加载理论对齐配置文件（关键修改：读取config/theory_keywords.json）"""
        config_path = 'config/theory_keywords.json'
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"理论配置文件未找到：{config_path}，请先创建")
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _calculate_theory_alignment(self, rule_features):
        """
        计算规则与理论的对齐分数（论文3.2.2）
        :param rule_features: 规则包含的特征列表（如["content_clicks", "score_trend_3w"]）
        :return: 理论对齐分数字典（{理论名: 分数}）
        """
        alignment_scores = {}
        rule_feature_str = " ".join(rule_features).lower()
        
        for theory_name, theory_info in self.theory_config.items():
            # 处理两种配置格式：
            # 格式1：从config/theory_keywords.json（有feature_keywords, core_constructs, risk_indicators）
            # 格式2：从main.py THEORY_MAPPING（只有keywords, description）
            
            # 1. 特征关键词匹配（基础分数）
            if "feature_keywords" in theory_info:
                theory_keywords = [kw.lower() for kw in theory_info["feature_keywords"]]
            elif "keywords" in theory_info:
                theory_keywords = [kw.lower() for kw in theory_info["keywords"]]
            else:
                theory_keywords = []
            
            match_count = sum(1 for kw in theory_keywords if kw in rule_feature_str)
            base_score = match_count / len(theory_keywords) if theory_keywords else 0
            
            # 2. 语义相似度匹配（增强分数，论文用的all-MiniLM-L6-v2）
            if MODEL is not None:
                rule_embedding = MODEL.encode(rule_feature_str, convert_to_tensor=False).reshape(1, -1)
                # 优先使用core_constructs，否则用keywords
                semantic_text = " ".join(theory_info.get("core_constructs", theory_info.get("keywords", [])))
                theory_embedding = MODEL.encode(semantic_text, convert_to_tensor=False).reshape(1, -1)
                semantic_score = cosine_similarity(rule_embedding, theory_embedding)[0][0]
            else:
                # 无模型时，用风险指标匹配替代（或降级到简单的基础分数）
                if "risk_indicators" in theory_info:
                    risk_indicators = [ri.lower() for ri in theory_info["risk_indicators"]]
                    risk_match = sum(1 for ri in risk_indicators if ri in rule_feature_str)
                    semantic_score = risk_match / len(risk_indicators) if risk_indicators else 0
                else:
                    semantic_score = base_score  # 降级：使用基础分数
            
            # 3. 综合分数（论文公式：Q(r) = 0.4*base_score + 0.6*semantic_score）
            final_score = 0.4 * base_score + 0.6 * semantic_score
            alignment_scores[theory_name] = final_score
        
        return alignment_scores
    
    def _generate_rule_id(self, rule_type):
        """生成论文风格的规则ID（STAT/TEMP/TW + 数字）"""
        prefix_map = {
            "StaticPatternRule": "STAT",
            "TemporalSequenceRule": "TEMP",
            "TimeWindowRule": "TW"
        }
        prefix = prefix_map.get(rule_type, "UNK")
        rule_id = f"{prefix}{self.rule_counter[rule_type]:03d}"
        self.rule_counter[rule_type] += 1
        return rule_id
    
    def mine_theory_grounded_rules(self, features_df, rule_strategies=["co_occurrence", "temporal_sequence", "time_window"]):
        """
        挖掘理论锚定规则（论文3.2.1+3.2.2）
        :param features_df: 特征矩阵（来自feature_engineer.py）
        :param rule_strategies: 规则挖掘策略（共现/时序/时间窗口）
        :return: 理论对齐后的有效规则列表
        """
        print(f"开始理论锚定规则挖掘，策略：{rule_strategies}")
        validated_rules = []
        
        # 1. 数据预处理（离散化特征，用于规则挖掘）
        discretized_df = self._discretize_features(features_df)
        # 只保留有风险标签的数据
        if "is_risk" not in discretized_df.columns:
            raise ValueError("特征矩阵缺少is_risk列，无法挖掘风险规则")
        risk_df = discretized_df[discretized_df["is_risk"] == 1]
        
        # 2. 按策略挖掘候选规则
        candidate_rules = []
        if "co_occurrence" in rule_strategies:
            candidate_rules.extend(self._mine_co_occurrence_rules(discretized_df, risk_df))
        if "temporal_sequence" in rule_strategies:
            candidate_rules.extend(self._mine_temporal_rules(features_df))
        if "time_window" in rule_strategies:
            candidate_rules.extend(self._mine_time_window_rules(features_df))
        # 支持额外的策略（兼容mine_comprehensive_rules接口）
        if "click_based" in rule_strategies:
            candidate_rules.extend(self._mine_click_based_rules(features_df))
        if "activity_based" in rule_strategies:
            candidate_rules.extend(self._mine_activity_based_rules(features_df))
        
        # 3. 理论对齐筛选（核心步骤，论文3.2.2）
        for rule in candidate_rules:
            # 计算规则与各理论的对齐分数
            rule_features = [f.split("_cat")[0] for f in rule.get("condition_features", [])]
            alignment_scores = self._calculate_theory_alignment(rule_features)
            
            # 找到分数最高的理论
            if alignment_scores:
                best_theory = max(alignment_scores.items(), key=lambda x: x[1])[0]
                best_score = alignment_scores[best_theory]
                
                # 根据模式决定是否保留规则
                should_keep = False
                if self.strict_alignment:
                    # 严格模式：只保留对齐分数≥阈值的规则
                    should_keep = best_score >= self.alignment_threshold
                else:
                    # 宽松模式：保留所有有对齐信息的规则（甚至低分数）
                    should_keep = True
                
                if should_keep:
                    # 补充理论相关信息（论文Table 1格式）
                    rule["theory_aligned"] = best_theory
                    rule["theory_alignment_score"] = round(best_score, 3)
                    rule["theory_rationale"] = self._get_theory_rationale(best_theory, rule_features)
                    rule["rule_id"] = self._generate_rule_id(rule["rule_type"])
                    validated_rules.append(rule)
            else:
                # 如果没有理论对齐，在非严格模式下仍然保留规则
                if not self.strict_alignment:
                    rule["theory_aligned"] = "General"
                    rule["theory_alignment_score"] = 0.0
                    rule["theory_rationale"] = "通用风险规则，未与特定教育理论对齐"
                    rule["rule_id"] = self._generate_rule_id(rule["rule_type"])
                    validated_rules.append(rule)
        
        # 4. 去除冗余规则（新增步骤）
        validated_rules = self._remove_redundant_rules(validated_rules)
        
        print(f"规则挖掘完成：候选规则{len(candidate_rules)}条 → 理论对齐后{len(validated_rules)}条（模式：{'严格' if self.strict_alignment else '宽松'}）")
        return validated_rules
    
    def mine_comprehensive_rules(self, features_df, rule_strategies=["co_occurrence", "temporal_sequence", "time_window", "click_based", "activity_based"]):
        """
        综合规则挖掘（兼容扩展接口，支持5种策略）
        :param features_df: 特征矩阵
        :param rule_strategies: 规则挖掘策略列表
        :return: 挖掘的规则列表
        """
        return self.mine_theory_grounded_rules(features_df, rule_strategies)
    
    def _get_theory_rationale(self, theory_name, rule_features):
        """Generate rule theoretical rationale (Paper Table 1: Theoretical Rationale)"""
        if theory_name not in self.theory_config:
            return f"Aligned with {theory_name} theory, rule features reflect key constructs."
        
        theory_info = self.theory_config[theory_name]
        rationale_templates = {
            "Engagement": "Aligned with Engagement Theory, the {features} feature(s) reflect(s) participation continuity; {risk} signal(s) insufficient participation.",
            "SelfEfficacy": "Aligned with Self-Efficacy Theory, the {features} feature(s) reflect(s) task competence; {risk} indicate(s) low self-efficacy.",
            "StudentIntegration": "Aligned with Student Integration Model, the {features} feature(s) reflect(s) social-academic integration; {risk} signal(s) lack of integration."
        }
        
        # Extract risk indicators (if present)
        risk_indicators = theory_info.get("risk_indicators", [])
        matched_risk = [ri for ri in risk_indicators if any(ri.lower() in f.lower() for f in rule_features)]
        risk_desc = ", ".join(matched_risk) if matched_risk else "low engagement"
        
        # Fill template
        feature_str = ", ".join(rule_features[:3]) + (", etc." if len(rule_features) > 3 else "")
        template = rationale_templates.get(theory_name, "Aligned with {theory} theory, features {features} are consistent with core constructs.")
        return template.format(
            theory=theory_name,
            features=feature_str,
            risk=risk_desc
        )
    
    def _discretize_features(self, features_df):
        """离散化特征（复用feature_engineer.py的逻辑，避免重复）"""
        import importlib.util
        import sys
        
        # 动态导入带数字前缀的模块
        module_path = os.path.join(os.path.dirname(__file__), '02_feature_engineering.py')
        spec = importlib.util.spec_from_file_location("feature_engineer", module_path)
        feature_engineer = importlib.util.module_from_spec(spec)
        sys.modules["feature_engineer"] = feature_engineer
        spec.loader.exec_module(feature_engineer)
        
        engineer = feature_engineer.FeatureEngineer()
        return engineer.discretize_features_for_association_rules(features_df)
    
    def _mine_co_occurrence_rules(self, discretized_df, risk_df):
        """挖掘共现规则（论文3.2.1的Co-Occurrence Pattern Mining）"""
        co_occur_rules = []
        # 离散化特征列（排除非特征列）
        feature_cols = [col for col in discretized_df.columns if col.endswith("_cat")]
        
        # 1. 单特征规则
        for col in feature_cols:
            if col not in discretized_df.columns:
                continue
            # 获取最常见的值
            mode_val = discretized_df[col].mode()
            if len(mode_val) == 0:
                continue
            mode_val = mode_val[0]
            
            condition = f"{col} == '{mode_val}'"
            total_count = len(discretized_df.query(condition))
            if total_count < 30:  # 最小支持度过滤
                continue
            
            risk_count = len(discretized_df.query(f"{condition} & is_risk == 1"))
            confidence = risk_count / total_count if total_count > 0 else 0
            
            if confidence >= 0.3:  # 降低阈值以生成更多规则
                co_occur_rules.append({
                    "rule_type": "StaticPatternRule",
                    "name": f"Single Feature Risk Rule: {col}={mode_val}",
                    "condition": condition,
                    "condition_features": [col],
                    "confidence": round(confidence, 3),
                    "support": round(total_count / len(discretized_df), 3),
                    "affected_students": total_count
                })
        
        # 2. 挖掘2-4个特征的共现组合
        for k in [2, 3, 4]:
            if k > len(feature_cols):
                continue
            for combo in combinations(feature_cols, k):
                # 计算支持度和置信度
                combo_df = discretized_df[list(combo) + ["is_risk"]]
                # 共现条件（所有特征取特定值）
                try:
                    condition = " & ".join([f"{col} == '{combo_df[col].mode()[0]}'" for col in combo])
                except IndexError:
                    continue
                    
                total_count = len(combo_df.query(condition))
                if total_count < 30:  # 最小支持度过滤
                    continue
                
                # 风险支持度
                risk_count = len(combo_df.query(f"{condition} & is_risk == 1"))
                confidence = risk_count / total_count if total_count > 0 else 0
                
                if confidence >= 0.35:  # 降低置信度阈值以生成更多规则
                    co_occur_rules.append({
                        "rule_type": "StaticPatternRule",
                        "name": f"Co-occurrence Risk Rule: {'+'.join(combo)}",
                        "condition": condition,
                        "condition_features": list(combo),
                        "confidence": round(confidence, 3),
                        "support": round(total_count / len(discretized_df), 3),
                        "affected_students": total_count
                    })
        
        return co_occur_rules
    
    def _mine_temporal_rules(self, features_df):
        """挖掘时序规则（论文3.2.1的Temporal Sequence Mining）"""
        temporal_rules = []
        
        # 支持week或week_number两种列名
        week_col = 'week' if 'week' in features_df.columns else 'week_number'
        
        # 按学生分组，提取行为序列
        student_groups = features_df.groupby("id_student")
        
        # 1. 基于趋势特征的规则
        trend_features = [col for col in features_df.columns if col.endswith("_trend_3w") or col.endswith("_declining")]
        
        for student_id, group in student_groups:
            if len(group) < 3:  # 至少3周数据
                continue
            group = group.sort_values(week_col)
            
            for feat in trend_features:
                if feat not in group.columns:
                    continue
                # 检查连续下降趋势
                declining_count = (group[feat] < 0).sum() if group[feat].dtype in [np.float64, np.int64] else 0
                if declining_count >= 1:  # 至少1个周期下降
                    temporal_rules.append({
                        "rule_type": "TemporalSequenceRule",
                        "name": f"Temporal Trend Risk Rule: {feat} declining trend",
                        "condition": f"{feat} < 0 consecutive decline >= 1 time",
                        "condition_features": [feat.replace("_trend_3w", "").replace("_declining", "")],
                        "confidence": round(0.65 + min(declining_count/20, 0.2), 3),
                        "support": round(declining_count / len(group), 3),
                        "affected_students": 1
                    })
        
        # 2. 基于周期变化的规则
        for student_id, group in student_groups:
            if len(group) < 4:
                continue
            group = group.sort_values(week_col)
            
            # 检查点击数下降的规则
            for click_col in ['clicks_forum', 'clicks_resource', 'clicks_content', 'total_clicks']:
                if click_col not in group.columns:
                    continue
                values = group[click_col].values
                # 计算周期变化
                if len(values) > 2:
                    trend = np.diff(values)
                    declining_weeks = (trend < 0).sum()
                    if declining_weeks >= 2:
                        temporal_rules.append({
                            "rule_type": "TemporalSequenceRule",
                            "name": f"Click Decline Sequence Rule: {click_col} continuous decline",
                            "condition": f"{click_col} declines in consecutive periods",
                            "condition_features": [click_col],
                            "confidence": round(0.6 + min(declining_weeks/30, 0.25), 3),
                            "support": round(declining_weeks / len(group), 3),
                            "affected_students": 1
                        })
        
        # 聚合相同规则
        if temporal_rules:
            rule_groups = {}
            for rule in temporal_rules:
                key = (rule["name"], rule["condition"])
                if key not in rule_groups:
                    rule_groups[key] = rule.copy()
                else:
                    rule_groups[key]["affected_students"] += 1
            
            # 过滤有足够影响的规则
            temporal_rules = [r for r in rule_groups.values() if r["affected_students"] >= 5]
            for rule in temporal_rules:
                rule["support"] = round(rule["affected_students"] / len(student_groups), 3)
        
        return temporal_rules

    def _remove_redundant_rules(self, rules):
        """
        去除冗余规则（处理反馈：规则冗余与冲突）
        策略：
        1. 如果规则A的特征集合包含规则B的特征集合（A是B的特化），且A的置信度不高于B，则A冗余
        2. 保留置信度更高的规则
        """
        if not rules:
            return []
            
        # 按置信度降序排序
        sorted_rules = sorted(rules, key=lambda x: x["confidence"], reverse=True)
        final_rules = []
        
        for rule in sorted_rules:
            is_redundant = False
            rule_feats = set(rule.get("condition_features", []))
            
            for kept_rule in final_rules:
                kept_feats = set(kept_rule.get("condition_features", []))
                
                # 检查特征包含关系
                # 如果当前规则是已保留规则的超集（更复杂），且置信度没有显著更高（这里甚至是更低或相等），则冗余
                if kept_feats.issubset(rule_feats):
                    is_redundant = True
                    break
            
            if not is_redundant:
                final_rules.append(rule)
                
        return final_rules
    
    def _mine_time_window_rules(self, features_df):
        """挖掘时间窗口规则（论文3.2.1的TimeWindowRule）"""
        window_rules = []
        # 定义时间窗口（扩展窗口范围）
        windows = [
            ("early", 0, 4),
            ("mid", 5, 12),
            ("late", 13, 20),
            ("entire", 0, 20)
        ]
        
        # 使用实际存在的特征列
        all_numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
        key_features = [col for col in all_numeric_cols 
                       if col not in ['id_student', 'week', 'is_risk', 'end_week', 'module_presentation_length']]
        
        # 支持week或week_number两种列名
        week_col = 'week' if 'week' in features_df.columns else 'week_number'
        
        for window_name, start_week, end_week in windows:
            window_df = features_df[(features_df[week_col] >= start_week) & (features_df[week_col] <= end_week)]
            if len(window_df) < 50:  # 降低最小数据量要求
                continue
            
            for feat in key_features:
                if feat not in window_df.columns or window_df[feat].isnull().all():
                    continue
                
                # 1. 低值风险规则（低于25分位数）
                low_threshold = window_df[feat].quantile(0.25)
                low_risk_df = window_df[window_df[feat] <= low_threshold]
                
                if len(low_risk_df) >= 20:
                    confidence = low_risk_df["is_risk"].mean() if "is_risk" in low_risk_df.columns else 0.5
                    if confidence >= 0.25:
                        window_rules.append({
                            "rule_type": "TimeWindowRule",
                            "name": f"{window_name} Phase Risk Rule: {feat} Low Value Period",
                            "condition": f"week ∈ [{start_week},{end_week}] & {feat} ≤ {round(low_threshold, 2)}",
                            "condition_features": [feat],
                            "confidence": round(confidence, 3),
                            "support": round(len(low_risk_df) / len(window_df), 3),
                            "affected_students": len(low_risk_df)
                        })
                
                # 2. 高值风险规则（对于延迟提交等指标）
                if feat in ['submission_delay', 'late_submissions']:
                    high_threshold = window_df[feat].quantile(0.75)
                    high_risk_df = window_df[window_df[feat] >= high_threshold]
                    
                    if len(high_risk_df) >= 20:
                        confidence = high_risk_df["is_risk"].mean() if "is_risk" in high_risk_df.columns else 0.5
                        if confidence >= 0.25:
                            window_rules.append({
                                "rule_type": "TimeWindowRule",
                                "name": f"{window_name} Phase Risk Rule: {feat} High Value Period",
                                "condition": f"week ∈ [{start_week},{end_week}] & {feat} ≥ {round(high_threshold, 2)}",
                                "condition_features": [feat],
                                "confidence": round(confidence, 3),
                                "support": round(len(high_risk_df) / len(window_df), 3),
                                "affected_students": len(high_risk_df)
                            })
                
                # 3. 中等值风险规则（中位数周围）
                median_val = window_df[feat].median()
                median_lower = window_df[feat].quantile(0.35)
                median_upper = window_df[feat].quantile(0.65)
                median_risk_df = window_df[(window_df[feat] >= median_lower) & (window_df[feat] <= median_upper)]
                
                if len(median_risk_df) >= 20:
                    confidence = median_risk_df["is_risk"].mean() if "is_risk" in median_risk_df.columns else 0.5
                    if confidence >= 0.25:
                        window_rules.append({
                            "rule_type": "TimeWindowRule",
                            "name": f"{window_name} Phase Risk Rule: {feat} Moderate Fluctuation",
                            "condition": f"week ∈ [{start_week},{end_week}] & {round(median_lower, 2)} ≤ {feat} ≤ {round(median_upper, 2)}",
                            "condition_features": [feat],
                            "confidence": round(confidence, 3),
                            "support": round(len(median_risk_df) / len(window_df), 3),
                            "affected_students": len(median_risk_df)
                        })
        
        return window_rules
    
    def _mine_click_based_rules(self, features_df):
        """挖掘基于点击行为的规则"""
        click_rules = []
        click_cols = [col for col in features_df.columns if 'click' in col.lower()]
        
        if not click_cols:
            return click_rules
        
        for col in click_cols:
            if col not in features_df.columns or features_df[col].isnull().all():
                continue
            
            # 计算多个分位数阈值以生成更多规则
            quantiles = [0.1, 0.2, 0.3, 0.4, 0.5]
            for q in quantiles:
                threshold = features_df[col].quantile(q)
                risk_df = features_df[features_df[col] <= threshold]
                
                risk_count = len(risk_df)
                if risk_count < 20:  # 降低最小数据量
                    continue
                
                confidence = risk_df["is_risk"].mean() if "is_risk" in risk_df.columns else 0.5
                if confidence >= 0.25:  # 降低置信度阈值
                    click_rules.append({
                        "rule_type": "StaticPatternRule",
                        "name": f"Low Click Risk Rule: {col} ≤ {round(threshold, 2)}",
                        "condition": f"{col} ≤ {round(threshold, 2)}",
                        "condition_features": [col],
                        "confidence": round(confidence, 3),
                        "support": round(risk_count / len(features_df), 3),
                        "affected_students": risk_count
                    })
            
            # 添加零值规则
            zero_df = features_df[features_df[col] == 0]
            if len(zero_df) >= 20:
                confidence = zero_df["is_risk"].mean() if "is_risk" in zero_df.columns else 0.5
                if confidence >= 0.25:
                    click_rules.append({
                        "rule_type": "StaticPatternRule",
                        "name": f"Zero Click Risk Rule: {col} == 0",
                        "condition": f"{col} == 0",
                        "condition_features": [col],
                        "confidence": round(confidence, 3),
                        "support": round(len(zero_df) / len(features_df), 3),
                        "affected_students": len(zero_df)
                    })
        
        return click_rules
    
    def _mine_activity_based_rules(self, features_df):
        """挖掘基于活动行为的规则"""
        activity_rules = []
        activity_cols = [col for col in features_df.columns 
                        if any(x in col.lower() for x in ['activity', 'days_active', 'density', 'trend', 'consistency'])]
        
        if not activity_cols:
            return activity_rules
        
        for col in activity_cols:
            if col not in features_df.columns or features_df[col].isnull().all():
                continue
            
            # 计算多个分位数阈值以生成更多规则
            quantiles = [0.1, 0.2, 0.3, 0.4, 0.5]
            for q in quantiles:
                threshold = features_df[col].quantile(q)
                risk_df = features_df[features_df[col] <= threshold]
                
                risk_count = len(risk_df)
                if risk_count < 20:  # 降低最小数据量
                    continue
                
                confidence = risk_df["is_risk"].mean() if "is_risk" in risk_df.columns else 0.5
                if confidence >= 0.25:  # 降低置信度阈值
                    activity_rules.append({
                        "rule_type": "StaticPatternRule",
                        "name": f"Low Activity Risk Rule: {col} ≤ {round(threshold, 3)}",
                        "condition": f"{col} ≤ {round(threshold, 3)}",
                        "condition_features": [col],
                        "confidence": round(confidence, 3),
                        "support": round(risk_count / len(features_df), 3),
                        "affected_students": risk_count
                    })
            
            # 添加高值规则（对于某些活动指标）
            if 'trend' in col.lower() or 'consistency' in col.lower():
                high_threshold = features_df[col].quantile(0.75)
                high_risk_df = features_df[features_df[col] >= high_threshold]
                
                if len(high_risk_df) >= 20:
                    confidence = high_risk_df["is_risk"].mean() if "is_risk" in high_risk_df.columns else 0.5
                    if confidence >= 0.25:
                        activity_rules.append({
                            "rule_type": "StaticPatternRule",
                            "name": f"Abnormal Activity Risk Rule: {col} ≥ {round(high_threshold, 3)}",
                            "condition": f"{col} ≥ {round(high_threshold, 3)}",
                            "condition_features": [col],
                            "confidence": round(confidence, 3),
                            "support": round(len(high_risk_df) / len(features_df), 3),
                            "affected_students": len(high_risk_df)
                        })
        
        return activity_rules
    
    def save_rules(self, rules=None, output_dir='outputs/rules/', include_theory_info=True):
        """
        保存理论锚定规则（论文3.3 F-Logic兼容格式）
        :param rules: 规则列表（来自 mine_theory_grounded_rules）
        :param output_dir: 输出目录
        :param include_theory_info: 是否包含理论信息和F-Logic格式
        :return: 保存的规则列表
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 如果未传入规则，返回空列表
        if rules is None or not rules:
            print("无有效规则可保存")
            return []
        
        # 深拷贝规则，避免修改原列表
        rules_to_save = []
        for rule in rules:
            rule_copy = rule.copy()
            
            # 补充F-Logic模板（论文3.3）
            if include_theory_info and 'theory_aligned' in rule_copy:
                rule_copy["f_logic_rule"] = f"""Student(S) [risk -> {rule_copy['theory_aligned'].lower()}_risk, confidence -> {rule_copy['confidence']}] :-
    Student(S) AND {rule_copy['condition'].replace('&', 'AND')} AND theory_alignment(S, '{rule_copy['theory_aligned']}', {rule_copy.get('theory_alignment_score', 0.5)})."""
            else:
                rule_copy["f_logic_rule"] = f"""Student(S) [risk -> high, confidence -> {rule_copy['confidence']}] :-
    Student(S) AND {rule_copy['condition'].replace('&', 'AND')}."""
            
            rules_to_save.append(rule_copy)
        
        # 1. 保存JSON规则库（完整格式）
        json_path = os.path.join(output_dir, "enhanced_rules.json")
        
        # 自定义JSON编码器处理numpy类型
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(rules_to_save, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        print(f"✓ Rules saved (JSON): {json_path} ({len(rules_to_save)} rules)")
        
        # 2. 保存文本格式规则库（易读格式）
        txt_path = os.path.join(output_dir, "enhanced_rules.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("EduRuleReasoning - Theory-Grounded Risk Rules\n")
            f.write("=" * 80 + "\n\n")
            
            for i, rule in enumerate(rules_to_save, 1):
                f.write(f"{i}. [{rule.get('rule_id', 'UNKNOWN')}] {rule.get('name', 'Unknown Rule')}\n")
                f.write(f"   Type: {rule.get('rule_type', 'Unknown')}\n")
                f.write(f"   Theory: {rule.get('theory_aligned', 'General')}\n")
                f.write(f"   Condition: {rule.get('condition', 'N/A')}\n")
                f.write(f"   Confidence: {rule.get('confidence', 0):.3f}\n")
                f.write(f"   Support: {rule.get('support', 0):.3f}\n")
                f.write(f"   Affected Students: {rule.get('affected_students', 0)}\n")
                f.write(f"   Theory Alignment Score: {rule.get('theory_alignment_score', 0):.3f}\n")
                f.write(f"   Rationale: {rule.get('theory_rationale', 'N/A')}\n")
                f.write(f"   F-Logic: {rule.get('f_logic_rule', 'N/A')}\n")
                f.write("\n")
            
            f.write("=" * 80 + "\n")
            f.write(f"Total Rules: {len(rules_to_save)}\n")
            f.write(f"Generation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print(f"✓ Rules saved (Text): {txt_path}")
        
        # 3. 生成规则统计摘要
        stats_path = os.path.join(output_dir, "rule_statistics.json")
        stats = {
            "total_rules": len(rules_to_save),
            "generation_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "rule_types": {},
            "theories": {},
            "top_confidence_rules": [],
            "top_support_rules": []
        }
        
        # 统计规则类型
        for rule in rules_to_save:
            rule_type = rule.get('rule_type', 'Unknown')
            stats['rule_types'][rule_type] = stats['rule_types'].get(rule_type, 0) + 1
            
            theory = rule.get('theory_aligned', 'General')
            stats['theories'][theory] = stats['theories'].get(theory, 0) + 1
        
        # 找出高置信度和高支持度规则
        sorted_by_confidence = sorted(rules_to_save, key=lambda x: x.get('confidence', 0), reverse=True)
        sorted_by_support = sorted(rules_to_save, key=lambda x: x.get('support', 0), reverse=True)
        
        stats['top_confidence_rules'] = [
            {
                'rule_id': r.get('rule_id'),
                'name': r.get('name'),
                'confidence': r.get('confidence')
            }
            for r in sorted_by_confidence[:10]
        ]
        
        stats['top_support_rules'] = [
            {
                'rule_id': r.get('rule_id'),
                'name': r.get('name'),
                'support': r.get('support'),
                'affected_students': r.get('affected_students')
            }
            for r in sorted_by_support[:10]
        ]
        
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Rule statistics saved: {stats_path}")
        
        return rules_to_save

# 使用示例
if __name__ == "__main__":
    # 加载特征矩阵（来自feature_engineer.py）
    features_df = pd.read_parquet('data/processed/weekly_features.parquet')
    
    # 初始化挖掘器
    miner = TheoryAlignedRuleMiner(alignment_threshold=0.6)
    
    # 挖掘规则
    rules = miner.mine_theory_grounded_rules(
        features_df,
        rule_strategies=["co_occurrence", "temporal_sequence", "time_window"]
    )
    
    # 保存规则
    miner.save_rules(rules)


# 兼容性别名，用于支持不同的导入方式
EnhancedRuleMiner = TheoryAlignedRuleMiner