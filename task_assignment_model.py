import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from typing import List, Dict, Any

class AdvancedTaskAssignmentModel:
    def __init__(self):
        self.task_priority_model = None
        self.employee_recommendation_model = None
        self.task_complexity_model = None
    
    def prepare_task_data(self, tasks: List[Dict]) -> pd.DataFrame:
        task_data = []
        for task in tasks:
            task_entry = {
                'deadline': task['deadline'],
                'importance': task['importance'],
                'tech_stack_complexity': self._calculate_tech_complexity(task['tech_stack']),
                'max_team_size': task['max_team_size']
            }
            task_data.append(task_entry)
        return pd.DataFrame(task_data)
    
    def _calculate_tech_complexity(self, tech_stack: List[str]) -> float:
        complexity_map = {
            'AI': 9, 'ML': 8, 'Cloud': 7, 
            'Backend': 6, 'Frontend': 5, 
            'Database': 4, 'Testing': 3
        }
        return np.mean([complexity_map.get(tech, 5) for tech in tech_stack])
    
    def train_task_priority_model(self, tasks: List[Dict]):
        # Prepare data
        task_df = self.prepare_task_data(tasks)
        
        # Add synthetic priority labels for training
        task_df['priority'] = np.where(
            (task_df['importance'] > 7) & (task_df['deadline'] < 5), 
            'High',
            np.where(
                (task_df['importance'] > 5) & (task_df['deadline'] < 10), 
                'Medium', 
                'Low'
            )
        )
        
        # Preprocessing and model pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]), ['deadline', 'importance', 'tech_stack_complexity', 'max_team_size'])
            ])
        
        self.task_priority_model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', GradientBoostingClassifier(
                n_estimators=200, 
                learning_rate=0.1, 
                max_depth=4
            ))
        ])
        
        # Split and train
        X = task_df[['deadline', 'importance', 'tech_stack_complexity', 'max_team_size']]
        y = task_df['priority']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
        
        self.task_priority_model.fit(X_train, y_train)
        
        # Evaluation
        y_pred = self.task_priority_model.predict(X_test)
        print(classification_report(y_test, y_pred))
    
    def recommend_employees(self, task: Dict, employees: List[Dict]) -> List[int]:
        # Advanced employee recommendation logic
        employee_scores = []
        
        for employee in employees:
            skill_match = len(set(task['tech_stack']) & set(employee['skills'])) / len(task['tech_stack'])
            workload_factor = 1 / (1 + len(employee['current_tasks']))
            
            score = (
                skill_match * 0.6 + 
                workload_factor * 0.4
            )
            
            employee_scores.append({
                'id': employee['id'],
                'score': score
            })
        
        # Sort and return top N employee IDs
        return [
            emp['id'] for emp in 
            sorted(employee_scores, key=lambda x: x['score'], reverse=True)
        ][:task['max_team_size']]

def main():
    # Example usage
    model = AdvancedTaskAssignmentModel()
    
    # Simulated tasks and employees for demonstration
    tasks = [
        {
            'deadline': 10,
            'importance': 8,
            'tech_stack': ['AI', 'ML'],
            'max_team_size': 3
        }
    ]
    
    employees = [
        {
            'id': 101,
            'skills': ['Python', 'AI', 'ML'],
            'current_tasks': []
        },
        {
            'id': 102,
            'skills': ['Java', 'Backend'],
            'current_tasks': []
        }
    ]
    
    # Train model
    model.train_task_priority_model(tasks)
    
    # Recommend employees
    recommended_employees = model.recommend_employees(tasks[0], employees)
    print("Recommended Employees:", recommended_employees)

if __name__ == "__main__":
    main()
