import uuid
from datetime import datetime, timedelta
from model_utils import TaskPriorityModel, TeamCompatibilityAnalyzer, CompletionPredictor

class TaskManager:
    def __init__(self):
        self.tasks = []
        self.employees_df = self.initialize_employees()
        self.priority_model = TaskPriorityModel()
    
    @staticmethod
    def initialize_employees():
        # Sample employee data - in production, this would come from a database
        employee_data = {
            'Employee ID': [101, 102, 103, 104, 105],
            'Name': ['John Doe', 'Jane Smith', 'Bob Wilson', 'Alice Brown', 'Charlie Davis'],
            'Skill Set': [
                ['Python', 'ML', 'AI'],
                ['Java', 'Backend', 'Database'],
                ['Python', 'AI', 'Data Science'],
                ['JavaScript', 'Frontend', 'React'],
                ['Python', 'Database', 'Backend']
            ],
            'Current Workload': [3, 5, 2, 4, 1],
            'Performance Score': [8.5, 7.5, 9.0, 8.0, 7.0],
            'Team': ['ML Team', 'Backend Team', 'ML Team', 'Frontend Team', 'Backend Team']
        }
        return pd.DataFrame(employee_data)
    
    def create_task(self, name, deadline, tech_stack, importance, dependencies=0):
        task = {
            'Task ID': str(uuid.uuid4()),
            'Name': name,
            'Created': datetime.now(),
            'Deadline': deadline,
            'Tech Stack': tech_stack,
            'Importance': importance,
            'Dependencies': dependencies,
            'Complexity': calculate_task_complexity(tech_stack, dependencies, deadline),
            'Assigned To': None,
            'Status': 'Pending'
        }
        self.tasks.append(task)
        return task
    
    def get_team_recommendation(self, task):
        if not self.needs_team(task):
            return None
        
        available_employees = self.employees_df[self.employees_df['Current Workload'] < 8]
        best_team = None
        best_score = 0
        
        # Try different team sizes
        for team_size in range(2, min(4, len(available_employees) + 1)):
            for team in combinations(available_employees['Employee ID'], team_size):
                team_score = self.evaluate_team(team, task)
                if team_score > best_score:
                    best_score = team_score
                    best_team = team
        
        return best_team, best_score
    
    def needs_team(self, task):
        return (
            task['Complexity'] > 7 and
            task['Deadline'] < 5 and
            task['Importance'] > 7
        )
    
    def evaluate_team(self, team, task):
        compatibility = TeamCompatibilityAnalyzer.calculate_compatibility(team, self.employees_df)
        completion_prob = CompletionPredictor.predict_team(task, team, self.employees_df)
        return (compatibility * 0.4 + completion_prob * 0.6)
    
    def assign_task(self, task_id, assignee_ids):
        task = next((t for t in self.tasks if t['Task ID'] == task_id), None)
        if not task:
            return False
        
        task['Assigned To'] = assignee_ids
        for emp_id in assignee_ids:
            self.employees_df.loc[
                self.employees_df['Employee ID'] == emp_id, 
                'Current Workload'
            ] += 1
        
        return True
    
    def display_task_details(self, task_id):
        task = next((t for t in self.tasks if t['Task ID'] == task_id), None)
        if not task:
            return
        
        print("\n=== Task Details ===")
        print(f"Name: {task['Name']}")
        print(f"Deadline: {task['Deadline']} days")
        print(f"Complexity: {task['Complexity']}/10")
        print(f"Tech Stack: {', '.join(task['Tech Stack'])}")
        
        if isinstance(task['Assigned To'], list):
            print("\nAssigned Team:")
            for emp_id in task['Assigned To']:
                emp = self.employees_df[self.employees_df['Employee ID'] == emp_id].iloc[0]
                print(f"\nMember: {emp['Name']}")
                print(f"Skills: {', '.join(emp['Skill Set'])}")
                print(f"Current Workload: {emp['Current Workload']}/10")
        elif task['Assigned To']:
            emp = self.employees_df[self.employees_df['Employee ID'] == task['Assigned To']].iloc[0]
            print(f"\nAssigned To: {emp['Name']}")
            print(f"Skills: {', '.join(emp['Skill Set'])}")
            print(f"Current Workload: {emp['Current Workload']}/10")

def main():
    manager = TaskManager()
    
    while True:
        print("\n=== Task Management System ===")
        print("1. Create New Task")
        print("2. Assign Task")
        print("3. View Tasks")
        print("4. Exit")
        
        choice = input("\nEnter your choice: ")
        
        if choice == '1':
            name = input("Task Name: ")
            deadline = int(input("Deadline (days): "))
            tech_stack = input("Required Skills (comma-separated): ").split(',')
            importance = int(input("Importance (1-10): "))
            dependencies = int(input("Number of Dependencies: "))
            
            task = manager.create_task(name, deadline, tech_stack, importance, dependencies)
            print(f"\nTask created with ID: {task['Task ID']}")
        
        elif choice == '2':
            task_id = input("Enter Task ID: ")
            task = next((t for t in manager.tasks if t['Task ID'] == task_id), None)
            
            if task:
                if manager.needs_team(task):
                    print("\nThis task is recommended for team assignment.")
                    print("1. Use AI recommended team")
                    print("2. Select team manually")
                    team_choice = input("Enter choice: ")
                    
                    if team_choice == '1':
                        team, score = manager.get_team_recommendation(task)
                        print(f"\nRecommended Team (Score: {score:.2f}):")
                        for member_id in team:
                            emp = manager.employees_df[
                                manager.employees_df['Employee ID'] == member_id
                            ].iloc[0]
                            print(f"- {emp['Name']} ({', '.join(emp['Skill Set'])})")
                        
                        if input("\nAccept recommendation? (y/n): ").lower() == 'y':
                            manager.assign_task(task_id, team)
                    else:
                        print("\nAvailable Employees:")
                        for _, emp in manager.employees_df.iterrows():
                            print(f"{emp['Employee ID']}: {emp['Name']} - {emp['Team']}")
                        
                        team = input("Enter employee IDs (comma-separated): ").split(',')
                        team = [int(id.strip()) for id in team]
                        manager.assign_task(task_id, team)
                else:
                    print("\nAvailable Employees:")
                    for _, emp in manager.employees_df.iterrows():
                        print(f"{emp['Employee ID']}: {emp['Name']} - {emp['Team']}")
                    
                    emp_id = int(input("Enter employee ID: "))
                    manager.assign_task(task_id, [emp_id])
            
        elif choice == '3':
            for task in manager.tasks:
                manager.display_task_details(task['Task ID'])
        
        elif choice == '4':
            break

if __name__ == "__main__":
    main()
