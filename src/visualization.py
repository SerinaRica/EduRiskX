# src/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

plt.rcParams['font.sans-serif'] = ['Arial']  # 避免中文乱码
plt.rcParams['axes.unicode_minus'] = False

class TheoryAlignedVisualizer:
    def __init__(self, theory_mapping=None):
        self.theory_mapping = theory_mapping or {
            "Engagement": "Behavioral Engagement",
            "SelfEfficacy": "Self-Efficacy",
            "StudentIntegration": "Student Integration"
        }
    
    def create_comprehensive_visualizations(self, features_df, rules, output_dir='outputs/visualizations/', include_theory_analysis=True):
        """创建所有可视化（含理论对齐分析）"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 基础可视化
        self.plot_risk_distribution(features_df, output_dir)
        self.plot_feature_correlation(features_df, output_dir)
        
        # 理论对齐相关可视化（核心补充）
        if include_theory_analysis and rules:
            self.plot_rule_theory_alignment(rules, output_dir)
            self.plot_rule_type_by_theory(rules, output_dir)
            self.plot_theory_risk_indicators(features_df, rules, output_dir)
        
        print(f"所有可视化已保存到：{output_dir}")
    
    def plot_risk_distribution(self, features_df, output_dir):
        """风险分布直方图"""
        plt.figure(figsize=(10, 6))
        risk_counts = features_df["is_risk"].value_counts()
        sns.barplot(x=risk_counts.index, y=risk_counts.values, palette=["#2ecc71", "#e74c3c"])
        plt.title("Student Dropout Risk Distribution", fontsize=14, fontweight='bold')
        plt.xlabel("Risk Label (0=No Risk, 1=High Risk)", fontsize=12)
        plt.ylabel("Number of Students", fontsize=12)
        plt.xticks([0, 1], ["No Risk", "High Risk"])
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "risk_distribution_en.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_feature_correlation(self, features_df, output_dir):
        """特征相关性热力图（只选核心特征）"""
        core_features = ["total_clicks", "content_clicks", "days_active", "avg_score", "late_submissions", "is_risk"]
        core_features = [f for f in core_features if f in features_df.columns]
        corr_df = features_df[core_features].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_df, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        plt.title("Core Feature Correlation Heatmap", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "feature_correlation_heatmap_en.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_rule_theory_alignment(self, rules, output_dir):
        """规则-理论对齐分数热力图（论文4.2风格）"""
        # 构建规则×理论分数矩阵
        rule_names = [f"{rule['rule_id']}: {rule['name'][:30]}..." for rule in rules]
        theories = list(self.theory_mapping.keys())
        alignment_matrix = np.zeros((len(rules), len(theories)))
        
        for i, rule in enumerate(rules):
            for j, theory in enumerate(theories):
                alignment_matrix[i, j] = rule.get("theory_alignment_score", 0) if rule["theory_aligned"] == theory else 0
        
        plt.figure(figsize=(12, len(rules)*0.5))
        sns.heatmap(alignment_matrix, annot=True, cmap="YlGnBu", fmt=".3f", 
                    xticklabels=[self.theory_mapping[t] for t in theories],
                    yticklabels=rule_names)
        plt.title("Rule-Theory Alignment Scores", fontsize=14, fontweight='bold')
        plt.xlabel("Educational Theories", fontsize=12)
        plt.ylabel("Rules", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "rule_theory_alignment_en.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_rule_type_by_theory(self, rules, output_dir):
        """按理论分组的规则类型分布（论文4.2风格）"""
        rule_data = []
        for rule in rules:
            rule_data.append({
                "theory": self.theory_mapping[rule["theory_aligned"]],
                "rule_type": rule["rule_type"].replace("Rule", ""),
                "confidence": rule["confidence"]
            })
        rule_df = pd.DataFrame(rule_data)
        
        plt.figure(figsize=(12, 6))
        sns.countplot(x="theory", hue="rule_type", data=rule_df, palette="Set2")
        plt.title("Rule Type Distribution by Educational Theory", fontsize=14, fontweight='bold')
        plt.xlabel("Educational Theories", fontsize=12)
        plt.ylabel("Number of Rules", fontsize=12)
        plt.legend(title="Rule Type")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "rule_type_by_theory_en.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_theory_risk_indicators(self, features_df, rules, output_dir):
        """各理论风险指标分布"""
        # 提取各理论的核心风险特征
        theory_features = {
            "Engagement": ["total_clicks", "content_clicks", "days_active"],
            "SelfEfficacy": ["avg_score", "late_submissions"],
            "StudentIntegration": ["forum_clicks"]
        }
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes = axes.flatten()
        
        for i, (theory, features) in enumerate(theory_features.items()):
            # 只选存在的特征
            valid_features = [f for f in features if f in features_df.columns]
            if not valid_features:
                axes[i].text(0.5, 0.5, "No Data", ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(self.theory_mapping[theory])
                continue
            
            # 按风险分组计算特征均值
            risk_0 = features_df[features_df["is_risk"] == 0][valid_features].mean()
            risk_1 = features_df[features_df["is_risk"] == 1][valid_features].mean()
            
            x = np.arange(len(valid_features))
            width = 0.35
            
            axes[i].bar(x - width/2, risk_0, width, label="No Risk", color="#2ecc71")
            axes[i].bar(x + width/2, risk_1, width, label="High Risk", color="#e74c3c")
            axes[i].set_title(self.theory_mapping[theory], fontsize=12, fontweight='bold')
            axes[i].set_xticks(x)
            axes[i].set_xticklabels(valid_features, rotation=45, ha='right')
            axes[i].legend()
        
        plt.suptitle("Risk Indicator Distribution by Educational Theory", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "theory_risk_indicators_en.png"), dpi=300, bbox_inches='tight')
        plt.close()