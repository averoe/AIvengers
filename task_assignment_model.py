import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from itertools import combinations

class TaskPriorityModel:
    def __init__(self):
        self.priority_clf = RandomForestClassifier(random_state=42)
        self.label_encoder = LabelEncoder()
        self.initialize_model()
    
    def initialize_model(self):
        task_data = {
            'Deadline': [5, 10, 3, 7, 1, 4, 6, 2, 8, 9, 5, 3, 7, 4, 2],
            'Complexity': [3, 7, 5, 4, 8, 6, 2, 9, 1, 10, 5, 7, 3, 6, 8],
            'Dependencies': [1, 2, 1, 3, 0, 2, 1, 0, 3, 2, 1, 2, 0, 3, 1],
            'Importance': [8, 5, 9, 6, 10, 7, 4, 8, 5, 9, 7, 6, 8, 5, 9],
            'Priority': ['High', 'Medium', 'High', 'Medium', 'High', 'Medium', 'Low', 'High', 'Low', 'High', 'Medium', 'High', 'Low', 'Medium', 'High']
        }
        
        task_df = pd.DataFrame(task_data)
        task_df['Priority'] = self.label_encoder.fit_transform(task_df['Priority'])
        
        X = task_df[['Deadline', 'Complexity', 'Dependencies', 'Importance']]
        y = task_df['Priority']
        
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        self.priority_clf.fit(X_train, y_train)
    
    def predict_priority(self, task_features):
        prediction = self.priority_clf.predict(task_features)
        return self.label_encoder.inverse_transform(prediction)[0]


class TeamCompatibilityAnalyzer:
    @staticmethod
    def calculate_compatibility(team_members, employees_df):
        if len(team_members) < 2:
            return 0
        
        compatibility_scores = []
        for mem1, mem2 in combinations(team_members, 2):
            emp1 = employees_df[employees_df['Employee ID'] == mem1].iloc[0]
            emp2 = employees_df[employees_df['Employee ID'] == mem2].iloc[0]
            
            # Skill complementarity
            shared_skills = len(set(emp1['Skill Set']) & set(emp2['Skill Set']))
            total_skills = len(set(emp1['Skill Set']) | set(emp2['Skill Set']))
            skill_diversity = (total_skills - shared_skills) / total_skills
            
            # Workload balance
            workload_diff = abs(emp1['Current Workload'] - emp2['Current Workload'])
            workload_balance = 1 - (workload_diff / 10)
            
            # Performance synergy
            performance_avg = (emp1['Performance Score'] + emp2['Performance Score']) / 20
            
            pair_score = (skill_diversity * 0.4 + workload_balance * 0.3 + performance_avg * 0.3) * 10
            compatibility_scores.append(pair_score)
        
        return sum(compatibility_scores) / len(compatibility_scores)

class CompletionPredictor:
    @staticmethod
    def predict_individual(task, employee, employees_df):
        emp_data = employees_df[employees_df['Employee ID'] == employee].iloc[0]
        
        # Calculate various factors
        skill_match = len(set(task['Tech Stack']) & set(emp_data['Skill Set'])) / len(task['Tech Stack'])
        workload_capacity = 1 - (emp_data['Current Workload'] / 10)
        performance_factor = emp_data['Performance Score'] / 10
        complexity_factor = 1 - (task['Complexity'] / 10)
        
        # Weighted probability calculation
        probability = (
            skill_match * 0.35 +
            workload_capacity * 0.25 +
            performance_factor * 0.25 +
            complexity_factor * 0.15
        ) * 100
        
        return round(min(100, probability), 2)
    
    @staticmethod
    def predict_team(task, team_members, employees_df):
        individual_probs = [CompletionPredictor.predict_individual(task, member, employees_df) 
                          for member in team_members]
        
        base_prob = sum(individual_probs) / len(individual_probs)
        team_size_bonus = min(20, len(team_members) * 5)
        compatibility_score = TeamCompatibilityAnalyzer.calculate_compatibility(team_members, employees_df)
        
        final_prob = base_prob + team_size_bonus + (compatibility_score * 2)
        return round(min(100, final_prob), 2)

# Utility functions for task complexity and skill matching
def calculate_task_complexity(tech_stack, dependencies, deadline):
    base_complexity = {
        'AI': 9, 'ML': 8, 'Backend': 7, 'Frontend': 6,
        'Database': 5, 'Testing': 4, 'Documentation': 3
    }
    
    tech_complexity = max([base_complexity.get(tech, 5) for tech in tech_stack])
    deadline_factor = max(1, 10 - deadline)
    dependency_factor = min(10, dependencies * 2)
    
    return round((tech_complexity * 0.4 + deadline_factor * 0.3 + dependency_factor * 0.3), 2)

def get_skill_match_score(required_skills, employee_skills):
    matched = set(required_skills) & set(employee_skills)
    return len(matched) / len(required_skills) if required_skills else 0
