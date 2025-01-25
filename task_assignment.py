import uuid
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from task_assignment_model import AdvancedTaskPriorityModel

class EmployeeManagementSystem:
    def __init__(self):
        self.employees = {}
        self.sick_leave_records = {}
        self.task_reallocation_log = []
    
    def add_employee(self, employee_id, name, skills, team):
        self.employees[employee_id] = {
            'id': employee_id,
            'name': name,
            'skills': skills,
            'team': team,
            'current_tasks': [],
            'sick_leave_history': []
        }
    
    def mark_sick_leave(self, employee_id, duration_days):
        if employee_id not in self.employees:
            logging.error(f"Employee {employee_id} not found")
            return False
        
        start_date = datetime.now()
        end_date = start_date + timedelta(days=duration_days)
        
        self.sick_leave_records[employee_id] = {
            'start_date': start_date,
            'end_date': end_date,
            'tasks_to_compensate': []
        }
        
        logging.info(f"{self.employees[employee_id]['name']} on sick leave")
        return True

class EnhancedTaskManager:
    def __init__(self):
        # Advanced logging setup
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s',
            filename='task_management.log'
        )
        
        # Core systems
        self.employee_system = EmployeeManagementSystem()
        self.priority_model = AdvancedTaskPriorityModel()
        
        # Task and allocation tracking
        self.tasks = []
        self.task_allocation_history = []
    
    def create_task(self, name, deadline, tech_stack, importance, max_team_size=3):
        task = {
            'id': str(uuid.uuid4()),
            'name': name,
            'deadline': deadline,
            'tech_stack': tech_stack,
            'importance': importance,
            'max_team_size': max_team_size,
            'assigned_employees': [],
            'status': 'Pending'
        }
        self.tasks.append(task)
        return task
    
    def allocate_task(self, task_id, employee_ids=None):
        task = next((t for t in self.tasks if t['id'] == task_id), None)
        if not task:
            logging.error(f"Task {task_id} not found")
            return False
        
        # If no specific employees provided, use AI recommendation
        if not employee_ids:
            employee_ids = self._recommend_employees(task)
        
        # Validate team size
        if len(employee_ids) > task['max_team_size']:
            logging.warning(f"Reducing team size to {task['max_team_size']}")
            employee_ids = employee_ids[:task['max_team_size']]
        
        # Allocation logic
        task['assigned_employees'] = employee_ids
        
        # Log allocation
        self.task_allocation_history.append({
            'task_id': task_id,
            'employees': employee_ids,
            'allocation_time': datetime.now()
        })
        
        logging.info(f"Task {task['name']} allocated to {len(employee_ids)} employees")
        return True
    
    def handle_sick_leave(self, employee_id, duration_days):
        # Mark employee sick
        self.employee_system.mark_sick_leave(employee_id, duration_days)
        
        # Reallocate their tasks
        employee_tasks = [
            task for task in self.tasks 
            if employee_id in task['assigned_employees']
        ]
        
        for task in employee_tasks:
            # Remove sick employee
            task['assigned_employees'].remove(employee_id)
            
            # Reallocate task
            new_employees = self._recommend_employees(task)
            task['assigned_employees'].extend(new_employees)
            
            # Log reallocation
            logging.info(f"Task {task['name']} reallocated due to employee sick leave")
    
    def _recommend_employees(self, task):
        # Placeholder for advanced employee recommendation
        # In real scenario, use ML-driven recommendation
        return [emp_id for emp_id in range(101, 106)][:task['max_team_size']]
    
    def interactive_cli(self):
        while True:
            print("\n=== Advanced Task Management System ===")
            print("1. Create Task")
            print("2. Allocate Task")
            print("3. Mark Employee Sick Leave")
            print("4. View Tasks")
            print("5. Exit")
            
            choice = input("Choose an option: ")
            
            if choice == '1':
                self._create_task_interaction()
            elif choice == '2':
                self._allocate_task_interaction()
            elif choice == '3':
                self._sick_leave_interaction()
            elif choice == '4':
                self._view_tasks()
            elif choice == '5':
                break
    
    def _create_task_interaction(self):
        name = input("Task Name: ")
        deadline = int(input("Deadline (days): "))
        tech_stack = input("Tech Stack (comma-separated): ").split(',')
        importance = int(input("Importance (1-10): "))
        
        task = self.create_task(name, deadline, tech_stack, importance)
        print(f"Task created with ID: {task['id']}")
    
    def _allocate_task_interaction(self):
        task_id = input("Enter Task ID: ")
        employee_ids = input("Employee IDs (comma-separated, or leave blank for AI recommendation): ")
        
        if employee_ids:
            employee_ids = [int(id.strip()) for id in employee_ids.split(',')]
            self.allocate_task(task_id, employee_ids)
        else:
            self.allocate_task(task_id)
    
    def _sick_leave_interaction(self):
        employee_id = int(input("Employee ID: "))
        duration = int(input("Sick Leave Duration (days): "))
        
        self.handle_sick_leave(employee_id, duration)
        print("Sick leave processed and tasks reallocated")
    
    def _view_tasks(self):
        for task in self.tasks:
            print(f"\nTask: {task['name']}")
            print(f"ID: {task['id']}")
            print(f"Assigned Employees: {task['assigned_employees']}")
            print(f"Status: {task['status']}")

def main():
    task_manager = EnhancedTaskManager()
    task_manager.interactive_cli()

if __name__ == "__main__":
    main()
