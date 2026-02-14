# src/05_rule_system_builder.py
import pandas as pd
import json
from typing import Dict, List, Any
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

class RuleSystemBuilder:
    def __init__(self):
        self.rules = {}
        self.rule_graph = nx.DiGraph()
        self.theory_hierarchy = {
            'root': ['self_efficacy', 'student_integration', 'engagement'],
            'self_efficacy': ['performance_avoidance', 'effort_withdrawal', 
                            'resource_avoidance'],
            'student_integration': ['academic_disengagement', 
                                   'social_isolation', 'community_lack'],
            'engagement': ['behavioral_decline', 'cognitive_superficial', 
                         'affective_negative']
        }
    
    def load_and_integrate_rules(self, rules_df: pd.DataFrame):
        """
        加载并整合所有挖掘出的规则
        """
        print(f"加载 {len(rules_df)} 条规则...")
        
        for _, rule in rules_df.iterrows():
            rule_id = rule['final_rule_id']
            
            # 提取规则信息
            rule_info = {
                'id': rule_id,
                'conditions': rule.get('conditions', rule.get('rule', '')),
                'prediction': rule.get('prediction', 'RISK'),
                'confidence': rule.get('confidence', rule.get('accuracy', 0.5)),
                'support': rule.get('support', 0.01),
                'theory_score': rule.get('theory_score', 0.0),
                'quality_score': rule.get('quality_score', 0.0),
                'rule_type': rule.get('rule_type', 'unknown'),
                'theory_alignment': {}
            }
            
            # 添加理论对齐分数
            for col in rule.index:
                if col.startswith('theory_'):
                    theory_name = col.replace('theory_', '')
                    rule_info['theory_alignment'][theory_name] = rule[col]
            
            # 确定主要归属的理论
            if rule_info['theory_alignment']:
                main_theory = max(rule_info['theory_alignment'].items(), 
                                 key=lambda x: x[1])[0]
                rule_info['main_theory'] = main_theory
            else:
                rule_info['main_theory'] = 'unknown'
            
            # 存储规则
            self.rules[rule_id] = rule_info
            
            # 添加到规则图
            self.rule_graph.add_node(rule_id, **rule_info)
    
    def build_inference_chains(self):
        """
        构建规则之间的推理链条
        例如：基础规则 -> 模式规则 -> 诊断规则
        """
        print("构建规则推理链条...")
        
        # 根据规则类型和理论分类
        type_categories = {
            '基础规则': ['total_clicks', 'forum_clicks', 'avg_score'],
            '模式规则': ['trend', 'declining', 'consecutive'],
            '诊断规则': ['RISK', 'self_efficacy', 'integration']
        }
        
        # 连接相关规则
        for rule_id, rule_info in self.rules.items():
            rule_text = rule_info['conditions'].lower()
            
            # 寻找可以作为前件的规则
            for other_id, other_info in self.rules.items():
                if rule_id != other_id:
                    other_prediction = other_info.get('prediction', '').lower()
                    
                    # 如果当前规则的条件包含其他规则的预测
                    if other_prediction and other_prediction in rule_text:
                        self.rule_graph.add_edge(other_id, rule_id, 
                                                relation='precedes')
        
        print(f"规则图构建完成: {self.rule_graph.number_of_nodes()} 节点, "
              f"{self.rule_graph.number_of_edges()} 边")
    
    def generate_interventions(self, activated_rules: List[str]) -> Dict:
        """
        根据激活的规则生成干预建议
        """
        interventions = {
            'self_efficacy': {
                'low': ['鼓励性反馈', '设定小目标', '展示成功案例'],
                'medium': ['技能训练工作坊', '一对一辅导', '调整任务难度'],
                'high': ['心理咨询转介', '学习计划重构', '休学建议']
            },
            'student_integration': {
                'low': ['邀请加入学习小组', '鼓励论坛发言', '组织线上社交活动'],
                'medium': ['分配学习伙伴', '增加教师互动', '创建兴趣社区'],
                'high': ['安排导师面谈', '参与校园活动', '改善学习环境']
            },
            'engagement': {
                'low': ['发送学习提醒', '提供学习计划模板', '增加互动内容'],
                'medium': ['个性化内容推荐', '游戏化学习元素', '定期进度反馈'],
                'high': ['学习动机访谈', '课程调整建议', '多模式学习支持']
            }
        }
        
        # 分析激活的规则
        theory_scores = {'self_efficacy': 0, 'student_integration': 0, 'engagement': 0}
        for rule_id in activated_rules:
            if rule_id in self.rules:
                rule_info = self.rules[rule_id]
                for theory, score in rule_info['theory_alignment'].items():
                    if theory in theory_scores:
                        theory_scores[theory] = max(theory_scores[theory], score)
        
        # 生成干预建议
        recommendations = []
        for theory, score in theory_scores.items():
            if score > 0.3:  # 阈值
                level = 'low' if score < 0.5 else 'medium' if score < 0.7 else 'high'
                theory_interventions = interventions[theory][level]
                recommendations.append({
                    'theory': theory,
                    'severity': level,
                    'score': score,
                    'interventions': theory_interventions
                })
        
        return {
            'risk_level': 'high' if len(activated_rules) >= 3 else 'medium' if len(activated_rules) >= 1 else 'low',
            'activated_rules': activated_rules,
            'theory_analysis': theory_scores,
            'recommendations': recommendations
        }
    
    def save_rule_system(self, output_dir='outputs/rules/'):
        """保存完整的规则系统"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 保存规则库为JSON
        rules_json = {}
        for rule_id, rule_info in self.rules.items():
            # 转换为可序列化的格式
            serializable_rule = {}
            for key, value in rule_info.items():
                if isinstance(value, (np.float32, np.float64)):
                    serializable_rule[key] = float(value)
                elif isinstance(value, np.ndarray):
                    serializable_rule[key] = value.tolist()
                else:
                    serializable_rule[key] = value
            rules_json[rule_id] = serializable_rule
        
        with open(f'{output_dir}/rule_library.json', 'w') as f:
            json.dump(rules_json, f, indent=2, ensure_ascii=False)
        
        # 2. 保存为CSV用于分析
        rules_df = pd.DataFrame.from_dict(self.rules, orient='index')
        rules_df.to_csv(f'{output_dir}/rules_detailed.csv', index_label='rule_id')
        
        # 3. 保存简化的规则集（用于部署）
        simple_rules = []
        for rule_id, rule_info in self.rules.items():
            if rule_info['quality_score'] > 0.4:  # 高质量规则
                simple_rules.append({
                    'rule_id': rule_id,
                    'condition': rule_info['conditions'],
                    'prediction': rule_info['prediction'],
                    'confidence': rule_info['confidence'],
                    'theory': rule_info.get('main_theory', 'unknown'),
                    'intervention_hint': self._generate_intervention_hint(rule_info)
                })
        
        simple_df = pd.DataFrame(simple_rules)
        simple_df.to_csv(f'{output_dir}/deployment_rules.csv', index=False)
        
        print(f"规则系统已保存到 {output_dir}")
        print(f"- 详细规则库: {len(rules_json)} 条规则")
        print(f"- 部署规则集: {len(simple_df)} 条高质量规则")
    
    def _generate_intervention_hint(self, rule_info):
        """根据规则生成干预提示"""
        theory = rule_info.get('main_theory', '')
        conditions = rule_info['conditions'].lower()
        
        hints = []
        
        if 'forum' in conditions and 'zero' in conditions:
            hints.append("鼓励参与在线讨论")
        
        if 'late' in conditions and 'submission' in conditions:
            hints.append("提供时间管理指导")
        
        if 'decline' in conditions and 'click' in conditions:
            hints.append("发送学习参与提醒")
        
        if 'score' in conditions and 'low' in conditions:
            hints.append("提供额外学习资源")
        
        # 基于理论
        if theory == 'self_efficacy':
            hints.append("增强学习信心的小任务")
        elif theory == 'student_integration':
            hints.append("促进同伴互动的活动")
        elif theory == 'engagement':
            hints.append("个性化学习路径调整")
        
        return '; '.join(set(hints)) if hints else "基于通用学习支持"
    
    def visualize_rule_network(self, output_path='outputs/visualizations/rule_network.png'):
        """可视化规则网络"""
        plt.figure(figsize=(15, 10))
        
        # 按理论分组着色
        color_map = {
            'self_efficacy': 'red',
            'student_integration': 'blue',
            'engagement': 'green',
            'unknown': 'gray'
        }
        
        # 为节点分配颜色
        node_colors = []
        for node in self.rule_graph.nodes():
            rule_info = self.rules.get(node, {})
            theory = rule_info.get('main_theory', 'unknown')
            node_colors.append(color_map.get(theory, 'gray'))
        
        # 绘制网络
        pos = nx.spring_layout(self.rule_graph, k=0.5, iterations=50)
        nx.draw_networkx_nodes(self.rule_graph, pos, 
                              node_color=node_colors,
                              node_size=300,
                              alpha=0.8)
        nx.draw_networkx_edges(self.rule_graph, pos, 
                              edge_color='gray',
                              alpha=0.3,
                              arrows=True,
                              arrowsize=10)
        nx.draw_networkx_labels(self.rule_graph, pos, 
                               font_size=8,
                               font_weight='bold')
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_patches = [Patch(color=color, label=theory) 
                         for theory, color in color_map.items()]
        plt.legend(handles=legend_patches, loc='upper left')
        
        plt.title(f"规则推理网络 ({self.rule_graph.number_of_nodes()} 条规则)", 
                 fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        
        # 保存图片
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"规则网络图已保存: {output_path}")