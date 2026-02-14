def map_interventions(triggered_rules):
    res = []
    seen = set()
    for r in triggered_rules:
        t = r.get("theory", "")
        if t == "Engagement":
            for s in ["Increase learning interaction activities", "Set weekly learning goals", "Provide learning progress feedback"]:
                if s not in seen:
                    res.append(s); seen.add(s)
        elif t == "SelfEfficacy":
            for s in ["Share success stories", "Break down complex tasks into small steps", "Give timely positive feedback"]:
                if s not in seen:
                    res.append(s); seen.add(s)
        elif t == "StudentIntegration":
            for s in ["Invite to join study groups", "Encourage participation in forum discussions", "Organize online social activities"]:
                if s not in seen:
                    res.append(s); seen.add(s)
    return res[:6]
