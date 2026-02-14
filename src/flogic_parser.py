# src/flogic_parser.py

import re
import json
from typing import Dict, List, Tuple, Any

class FLogicParser:
    """
    Parse and normalize F-Logic rules
    Handles rules in your JSON format
    """
    
    @staticmethod
    def extract_flogic_rules(json_rules: List[Dict]) -> List[Dict]:
        """
        Extract F-Logic rules from JSON
        Input: Your JSON rule list
        Output: Normalized F-Logic rule dictionary
        """
        normalized_rules = []
        
        for rule in json_rules:
            if 'f_logic_rule' not in rule:
                # Generate F-Logic rule if missing
                flogic_rule = FLogicParser._generate_flogic_rule(rule)
            else:
                flogic_rule = rule['f_logic_rule']
            
            # Parse F-Logic rule
            parsed = FLogicParser._parse_flogic_rule(flogic_rule, rule)
            
            normalized_rules.append({
                'rule_id': rule.get('rule_id'),
                'rule_type': rule.get('rule_type'),
                'name': rule.get('name'),
                'original_json': rule,
                'flogic_string': flogic_rule,
                'parsed': parsed,
                'theory': rule.get('theory_aligned'),
                'confidence': rule.get('confidence'),
                'theory_score': rule.get('theory_alignment_score')
            })
        
        return normalized_rules
    
    @staticmethod
    def _parse_flogic_rule(flogic_string: str, original_rule: Dict) -> Dict:
        """
        Parse F-Logic rule string into structured representation
        """
        # Remove newlines and extra spaces
        flogic_string = ' '.join(flogic_string.replace('\n', ' ').split())
        
        # Extract Head
        head_match = re.search(r'(.+?)\s*:-\s*(.+)', flogic_string)
        if not head_match:
            # Try other patterns if format differs
            return {'raw': flogic_string, 'conditions': []}
        
        head = head_match.group(1).strip()
        body = head_match.group(2).strip()
        
        # Parse Head
        head_parsed = FLogicParser._parse_head(head)
        
        # Parse Body (conditions)
        conditions = []
        body_parts = re.split(r'\s+AND\s+', body)
        
        for part in body_parts:
            condition = FLogicParser._parse_condition(part.strip())
            if condition:
                conditions.append(condition)
        
        return {
            'head': head_parsed,
            'conditions': conditions,
            'body_string': body
        }
    
    @staticmethod
    def _parse_head(head_string: str) -> Dict:
        """Parse rule head, e.g., Student(S)[risk -> selfefficacy_risk, confidence -> 0.438]"""
        match = re.match(r'(\w+)\((\w+)\)\s*\[(.+)\]', head_string)
        if match:
            object_type, variable, attributes = match.groups()
            
            # Parse attributes
            attrs = {}
            for attr in attributes.split(','):
                key_value = attr.split('->')
                if len(key_value) == 2:
                    key = key_value[0].strip()
                    value = key_value[1].strip()
                    attrs[key] = value
            
            return {
                'object_type': object_type,
                'variable': variable,
                'attributes': attrs
            }
        
        return {'raw': head_string}
    
    @staticmethod
    def _parse_condition(condition_string: str) -> Dict:
        """Parse individual condition"""
        # Handle various condition formats
        
        # 1. Theory alignment condition: theory_alignment(S, 'SelfEfficacy', 0.14399999380111694)
        theory_match = re.match(r'theory_alignment\((\w+),\s*\'([\w]+)\',\s*([\d.]+)\)', condition_string)
        if theory_match:
            var, theory, score = theory_match.groups()
            return {
                'type': 'theory_alignment',
                'variable': var,
                'theory': theory,
                'score': float(score)
            }
        
        # 2. Simple comparison: submission_delay ≥ 0.0
        comp_match = re.match(r'(\w+)\s*([<>]=?|[≠=]=)\s*([\d.\'\"\w]+)', condition_string)
        if comp_match:
            feature, operator, value = comp_match.groups()
            
            # Clean value
            if value.startswith("'") and value.endswith("'"):
                value = value[1:-1]
            elif value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            else:
                try:
                    value = float(value) if '.' in value else int(value)
                except ValueError:
                    pass  # Keep as string
            
            return {
                'type': 'comparison',
                'feature': feature,
                'operator': operator,
                'value': value
            }
        
        # 3. Range condition: week ∈ [0,4]
        range_match = re.match(r'(\w+)\s*∈\s*\[([\d.]+),\s*([\d.]+)\]', condition_string)
        if range_match:
            feature, lower, upper = range_match.groups()
            return {
                'type': 'range',
                'feature': feature,
                'lower': float(lower),
                'upper': float(upper)
            }
        
        # 4. Simple existence: Student(S)
        if re.match(r'\w+\(\w+\)', condition_string):
            return {
                'type': 'existence',
                'condition': condition_string
            }
        
        return {
            'type': 'raw',
            'condition': condition_string
        }
    
    @staticmethod
    def _generate_flogic_rule(rule: Dict) -> str:
        """Generate an F-Logic rule if missing in JSON"""
        rule_id = rule.get('rule_id', 'UNKNOWN')
        theory = rule.get('theory_aligned', 'Unknown')
        confidence = rule.get('confidence', 0.0)
        condition = rule.get('condition', '')
        
        # Simplified generation
        return f"Student(S)[risk -> {theory.lower()}_risk, confidence -> {confidence}] :- Student(S) AND {condition}."