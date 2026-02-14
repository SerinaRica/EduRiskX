def aggregate(rule_evidences, model_prob):
    # Calculate Rule Evidence Score (Max of triggered rules to capture severity)
    rule_score = 0.0
    if rule_evidences:
        # Use MAX score from rules. If a severe rule triggers, risk is high.
        # Default theory_score to 1.0 if not present to ensure rules have impact
        scores = [float(r.get("confidence", 0.0)) * float(r.get("theory_score", 1.0)) for r in rule_evidences]
        rule_score = max(scores) if scores else 0.0

    # Probabilistic OR (Union) logic
    # P(A or B) = P(A) + P(B) - P(A)*P(B)
    # This allows EITHER the neural model OR the rules to trigger a high belief
    final_belief = model_prob + rule_score - (model_prob * rule_score)
    
    return min(final_belief, 1.0)
