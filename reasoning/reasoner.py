from typing import Dict, Any, Tuple, List
from pathlib import Path
import numpy as np
from src.flogic_reasoner_optimized import OptimizedFLogicReasoner
from .evidence_aggregation import aggregate
from .intervention_mapper import map_interventions

class EduRuleReasoner:
    def __init__(self, rules_file: str = "outputs/rules/enhanced_rules.json", risk_threshold: float = 0.5):
        p = Path(rules_file)
        self.reasoner = OptimizedFLogicReasoner(str(p))
        self.risk_threshold = risk_threshold
    def reason(self, student_features: Dict[str, Any], risk_prob: float) -> Tuple[float, Dict[str, Any]]:
        # Always evaluate rules to ensure independent detection (OR logic)
        res = self.reasoner.evaluate_student(student_features)
        triggered = res["triggered_rules"]
        
        final_belief = aggregate(triggered, float(risk_prob))
        
        # Determine Severity/Urgency Level
        severity = "Low"
        if final_belief >= 0.8:
            severity = "Critical"
        elif final_belief >= 0.6:
            severity = "High"
        elif final_belief >= 0.4: # Assuming 0.5 is typical threshold, 0.4 is warning
            severity = "Medium"
            
        explanation = {
            "risk_prob": float(risk_prob), 
            "belief": float(final_belief), 
            "severity": severity,
            "triggered_count": len(triggered)
        }
        if triggered:
            by_theory = {}
            for r in triggered:
                t = r.get("theory", "N/A")
                by_theory.setdefault(t, []).append(r["rule_id"])
            explanation["theories"] = by_theory
            explanation["interventions"] = map_interventions(triggered)
        return float(final_belief), explanation
