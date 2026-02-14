# src/flogic_reasoner_optimized.py (修复版本)

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Set, Tuple
from .flogic_parser import FLogicParser

class OptimizedFLogicReasoner:
    """
    Optimized F-Logic Reasoning Engine
    Directly uses F-Logic rules from your JSON
    """
    
    def __init__(self, rules_file: str = "outputs/rules/enhanced_rules.json"):
        self.rules = self._load_rules(rules_file)
        self.parsed_rules = FLogicParser.extract_flogic_rules(self.rules)
        self.feature_index = self._build_feature_index()
        self.rule_by_id = {rule['rule_id']: rule for rule in self.parsed_rules}
        
        print(f"✓ Loaded {len(self.parsed_rules)} F-Logic rules")
        print(f"✓ Indexed {len(self.feature_index)} features")
    
    def _load_rules(self, file_path: str) -> List[Dict]:
        """Load JSON rules"""
        import json
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _build_feature_index(self) -> Dict[str, List[str]]:
        """Build feature index for faster reasoning - returns list of rule_ids instead of rule objects"""
        feature_index = {}
        
        for rule in self.parsed_rules:
            rule_id = rule['rule_id']
            conditions = rule['parsed']['conditions']
            
            for cond in conditions:
                if cond['type'] == 'comparison':
                    feature = cond['feature']
                    if feature not in feature_index:
                        feature_index[feature] = []
                    if rule_id not in feature_index[feature]:
                        feature_index[feature].append(rule_id)
                elif cond['type'] == 'range':
                    feature = cond['feature']
                    if feature not in feature_index:
                        feature_index[feature] = []
                    if rule_id not in feature_index[feature]:
                        feature_index[feature].append(rule_id)
        
        return feature_index
    
    def _check_condition(self, condition: Dict, student_data: Dict) -> bool:
        """检查单个条件是否满足"""
        cond_type = condition['type']
        
        if cond_type == 'comparison':
            return self._check_comparison(condition, student_data)
        elif cond_type == 'range':
            return self._check_range(condition, student_data)
        elif cond_type == 'theory_alignment':
            # 理论对齐条件在规则头部已处理，这里默认通过
            return True
        elif cond_type == 'existence':
            # 如 Student(S)，默认通过
            return True
        else:
            # 原始条件，尝试直接评估
            return self._evaluate_raw_condition(condition, student_data)
    
    def _check_comparison(self, condition: Dict, student_data: Dict) -> bool:
        """检查比较条件"""
        feature = condition['feature']
        operator = condition['operator']
        value = condition['value']
        
        if feature not in student_data:
            return False
        
        student_value = student_data[feature]
        
        # 处理字符串和数值比较
        if isinstance(value, str) and value.replace('.', '').replace('-', '').isdigit():
            value = float(value) if '.' in value else int(value)
        
        # 标准化操作符
        operator_map = {
            '≥': '>=',
            '≤': '<=',
            '≠': '!=',
            '=': '==',
            '>=': '>=',
            '<=': '<=',
            '!=': '!=',
            '==': '==',
            '>': '>',
            '<': '<'
        }
        
        op = operator_map.get(operator, operator)
        
        try:
            if op == '>=':
                return student_value >= value
            elif op == '<=':
                return student_value <= value
            elif op == '>':
                return student_value > value
            elif op == '<':
                return student_value < value
            elif op in ['==', '=']:
                return student_value == value
            elif op == '!=':
                return student_value != value
        except TypeError as e:
            # 如果类型不匹配，尝试转换
            try:
                student_value = float(student_value)
                if isinstance(value, str):
                    value = float(value)
                return self._check_comparison({
                    'feature': feature,
                    'operator': op,
                    'value': value
                }, {feature: student_value})
            except:
                return False
        
        return False
    
    def _check_range(self, condition: Dict, student_data: Dict) -> bool:
        """检查区间条件"""
        feature = condition['feature']
        lower = condition['lower']
        upper = condition['upper']
        
        if feature not in student_data:
            return False
        
        student_value = student_data[feature]
        return lower <= student_value <= upper
    
    def _evaluate_raw_condition(self, condition: Dict, student_data: Dict) -> bool:
        """评估原始条件字符串 - 简化实现"""
        cond_str = condition['condition']
        
        # 简单的字符串匹配
        if 'week ∈' in cond_str:
            # 尝试解析区间
            try:
                import re
                match = re.search(r'\[([\d.]+),\s*([\d.]+)\]', cond_str)
                if match:
                    lower, upper = float(match.group(1)), float(match.group(2))
                    if 'week' in student_data:
                        return lower <= student_data['week'] <= upper
            except:
                pass
        elif '≥' in cond_str or '<=' in cond_str or '==' in cond_str:
            # 尝试解析比较
            try:
                parts = cond_str.replace('≥', '>=').replace('≤', '<=').split()
                if len(parts) >= 3:
                    feature = parts[0]
                    operator = parts[1]
                    value_str = parts[2]
                    
                    if feature in student_data:
                        value = float(value_str) if '.' in value_str else int(value_str)
                        return self._check_comparison({
                            'feature': feature,
                            'operator': operator,
                            'value': value
                        }, student_data)
            except:
                pass
        
        return False
    
    def evaluate_student(self, student_data: Dict) -> Dict:
        """
        评估单个学生
        返回: 触发的规则和风险评估
        """
        student_id = student_data.get('id_student', 'unknown')
        
        # 只检查相关的规则 - 修复这里
        relevant_features = set(student_data.keys()) & set(self.feature_index.keys())
        candidate_rule_ids = set()
        
        for feature in relevant_features:
            for rule_id in self.feature_index.get(feature, []):
                candidate_rule_ids.add(rule_id)
        
        triggered_rules = []
        
        # 检查每个候选规则
        for rule_id in candidate_rule_ids:
            rule = self.rule_by_id.get(rule_id)
            if rule and self._check_rule(rule, student_data):
                triggered_rules.append({
                    'rule_id': rule['rule_id'],
                    'name': rule['name'],
                    'theory': rule['theory'],
                    'confidence': rule['confidence'],
                    'theory_score': rule['theory_score'],
                    'flogic_rule': rule['flogic_string']
                })
        
        # 证据合成
        overall_risk = self._synthesize_evidence(triggered_rules)
        
        # 生成解释
        explanation = self._generate_explanation(triggered_rules, overall_risk)
        
        return {
            'student_id': student_id,
            'triggered_rules': triggered_rules,
            'triggered_count': len(triggered_rules),
            'overall_risk': overall_risk,
            'explanation': explanation,
            'recommended_interventions': self._suggest_interventions(triggered_rules, overall_risk)
        }
    
    def _check_rule(self, rule: Dict, student_data: Dict) -> bool:
        """检查规则的所有条件是否都满足"""
        conditions = rule['parsed']['conditions']
        
        for condition in conditions:
            if not self._check_condition(condition, student_data):
                return False
        
        return True
    
        # src/flogic_reasoner_optimized.py (更新_synthesize_evidence方法)

    def _synthesize_evidence(self, triggered_rules: List) -> Dict:
        """
        合成证据 - 根据你的规则调整阈值
        """
        if not triggered_rules:
            return {
                'level': 'No Risk',
                'belief': 0.0,
                'confidence': 0.0,
                'theory_distribution': {}
            }
        
        # 按理论分组
        theory_scores = {}
        for rule in triggered_rules:
            theory = rule['theory']
            if theory not in theory_scores:
                theory_scores[theory] = []
            theory_scores[theory].append(rule['confidence'])
        
        # 计算每个理论的平均置信度
        theory_avg = {}
        for theory, scores in theory_scores.items():
            theory_avg[theory] = np.mean(scores)
        
        # 确定主导理论
        if theory_avg:
            dominant_theory = max(theory_avg.items(), key=lambda x: x[1])[0]
            dominant_score = theory_avg[dominant_theory]
        else:
            dominant_theory = None
            dominant_score = 0.0
        
        # 使用加权置信度（考虑规则数量和理论分数，降低放大系数以避免过度正例）
        weighted_confidence = dominant_score * (1 + 0.02 * len(triggered_rules))
        weighted_confidence = min(weighted_confidence, 1.0)  # 上限为1.0
        
        # 更保守的风险分级阈值（减少误报）
        if weighted_confidence >= 0.65:
            risk_level = 'High'
        elif weighted_confidence >= 0.55:
            risk_level = 'Medium'
        elif weighted_confidence >= 0.45:
            risk_level = 'Low'
        else:
            risk_level = 'No Risk'
        
        return {
            'level': risk_level,
            'belief': dominant_score,
            'weighted_confidence': weighted_confidence,
            'confidence': np.mean([r['confidence'] for r in triggered_rules]),
            'dominant_theory': dominant_theory,
            'theory_distribution': theory_avg,
            'rule_count': len(triggered_rules),
            'triggered_rule_ids': [r['rule_id'] for r in triggered_rules]
        }
    def _generate_explanation(self, triggered_rules: List, overall_risk: Dict) -> str:
        """生成可解释的输出"""
        if not triggered_rules:
            return "未触发任何风险规则，学生学习行为正常。"
        
        explanation = []
        explanation.append(f"🔍 风险评估: {overall_risk['level']} 风险")
        explanation.append(f"   置信度: {overall_risk['belief']:.1%}")
        explanation.append(f"   主导理论: {overall_risk.get('dominant_theory', 'N/A')}")
        explanation.append("")
        explanation.append("📋 触发的风险规则:")
        
        for i, rule in enumerate(triggered_rules[:5], 1):  # 最多显示5条
            explanation.append(f"   {i}. [{rule['rule_id']}] {rule['name']}")
            explanation.append(f"      理论: {rule['theory']} (对齐度: {rule['theory_score']:.3f})")
            explanation.append(f"      置信度: {rule['confidence']:.1%}")
        
        if len(triggered_rules) > 5:
            explanation.append(f"   ... 以及其他 {len(triggered_rules) - 5} 条规则")
        
        explanation.append("")
        explanation.append("🎯 风险指示器:")
        
        # 按理论分组显示
        theory_groups = {}
        for rule in triggered_rules:
            theory = rule['theory']
            if theory not in theory_groups:
                theory_groups[theory] = []
            theory_groups[theory].append(rule)
        
        for theory, rules in theory_groups.items():
            explanation.append(f"  • {theory} 理论指示器 ({len(rules)} 条规则):")
            for rule in rules[:2]:  # 每个理论最多显示2条
                rule_data = self.rule_by_id.get(rule['rule_id'])
                if rule_data:
                    conditions = []
                    for cond in rule_data['parsed']['conditions']:
                        if cond['type'] == 'comparison':
                            conditions.append(f"{cond['feature']} {cond['operator']} {cond['value']}")
                        elif cond['type'] == 'range':
                            conditions.append(f"{cond['feature']} ∈ [{cond['lower']}, {cond['upper']}]")
                    
                    if conditions:
                        explanation.append(f"    - {', '.join(conditions)}")
        
        return "\n".join(explanation)
    
    def _suggest_interventions(self, triggered_rules: List, overall_risk: Dict) -> List[str]:
        """根据触发的规则和建议干预"""
        interventions = []
        
        if overall_risk['level'] == 'High':
            interventions.append("🚨 需要立即干预")
            interventions.append("• 安排一对一学术辅导")
            interventions.append("• 联系学生了解学习困难")
        
        # 根据理论建议干预
        theory_interventions = {
            'Engagement': [
                "• 增加学习互动活动",
                "• 设置每周学习目标",
                "• 提供学习进度反馈"
            ],
            'SelfEfficacy': [
                "• 提供成功案例分享",
                "• 分解复杂任务为小步骤",
                "• 给予及时积极反馈"
            ],
            'StudentIntegration': [
                "• 邀请加入学习小组",
                "• 鼓励参与论坛讨论",
                "• 组织线上社交活动"
            ]
        }
        
        # 添加理论特定的干预
        for rule in triggered_rules:
            theory = rule['theory']
            if theory in theory_interventions:
                for intervention in theory_interventions[theory]:
                    if intervention not in interventions:
                        interventions.append(intervention)
        
        return interventions[:6]  # 最多返回6条建议
    
    def batch_evaluate(self, student_data_list: List[Dict]) -> Tuple[pd.DataFrame, List[Dict]]:
        """批量评估学生"""
        results = []
        
        for student_data in student_data_list:
            result = self.evaluate_student(student_data)
            results.append(result)
        
        # 转换为DataFrame
        df_results = pd.DataFrame([{
            'student_id': r['student_id'],
            'risk_level': r['overall_risk']['level'],
            'belief': r['overall_risk']['belief'],
            'confidence': r['overall_risk']['confidence'],
            'triggered_rules': len(r['triggered_rules']),
            'dominant_theory': r['overall_risk'].get('dominant_theory', 'None')
        } for r in results])
        
        return df_results, results
    
    def save_knowledge_base(self, output_dir: str = "outputs/knowledge_base"):
        """保存知识库文件"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 保存完整的F-Logic知识库
        flogic_file = os.path.join(output_dir, "complete_knowledge_base.flogic")
        with open(flogic_file, 'w', encoding='utf-8') as f:
            f.write("% Complete F-Logic Knowledge Base\n")
            f.write("% Generated from enhanced_rules.json\n\n")
            
            f.write("% Type Declarations\n")
            f.write("Student :: Object.\n")
            f.write("Rule :: Object.\n")
            f.write("Theory :: Object.\n\n")
            
            f.write("% All F-Logic Rules\n")
            for rule in self.parsed_rules:
                f.write(rule['flogic_string'])
                f.write("\n\n")
        
        print(f"✓ 知识库已保存: {flogic_file}")
        
        # 2. 保存规则统计
        stats_file = os.path.join(output_dir, "rule_statistics.json")
        stats = {
            'total_rules': len(self.parsed_rules),
            'rules_by_type': {},
            'rules_by_theory': {},
            'avg_confidence': float(np.mean([r['confidence'] for r in self.parsed_rules])),
            'avg_theory_score': float(np.mean([r['theory_score'] for r in self.parsed_rules]))
        }
        
        # 按类型统计
        for rule in self.parsed_rules:
            rule_type = rule['rule_type']
            stats['rules_by_type'][rule_type] = stats['rules_by_type'].get(rule_type, 0) + 1
            
            theory = rule['theory']
            stats['rules_by_theory'][theory] = stats['rules_by_theory'].get(theory, 0) + 1
        
        import json
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 规则统计已保存: {stats_file}")
        
        return flogic_file, stats_file
