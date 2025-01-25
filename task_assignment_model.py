import pandas as pd
import uuid
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import numpy as np

# Initialize an empty task database (list of dictionaries)
task_database = []

# Sample data for initial model training: 
# Task ID, Deadline (days), Complexity (1-10), Dependencies, Importance (1-10), Priority (label)
task_data = {
    'Task ID': [str(uuid.uuid4()) for _ in range(10)],  # Random Task IDs
    'Deadline': [5, 10, 3, 7, 1, 4, 6, 2, 8, 9],
    'Complexity': [3, 7, 5, 4, 8, 6, 2, 9, 1, 10],
    'Dependencies': [1, 2, 1, 3, 0, 2, 1, 0, 3, 2],
    'Importance': [8, 5, 9, 6, 10, 7, 4, 8, 5, 9],
    'Priority': ['High', 'Medium', 'High', 'Medium', 'High', 'Medium', 'Low', 'High', 'Low', 'High']
}

# Convert task data to DataFrame
task_df = pd.DataFrame(task_data)

# Encode the Priority labels
label_encoder = LabelEncoder()
task_df['Priority'] = label_encoder.fit_transform(task_df['Priority'])

# Features and labels for task priority prediction
X = task_df[['Deadline', 'Complexity', 'Dependencies', 'Importance']]
y = task_df['Priority']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ensure enough samples for SMOTE resampling
if X_train.shape[0] > 5:
    # Dynamically adjust k_neighbors based on minority class sample size
    minority_class_samples = y_train.value_counts().min()
    k_neighbors = min(3, minority_class_samples - 1)  
    
    # Only apply SMOTE if k_neighbors is at least 1
    if k_neighbors >= 1:
        smote = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=k_neighbors)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    else:
        X_resampled, y_resampled = X_train, y_train # Skip SMOTE if not enough samples
else:
    X_resampled, y_resampled = X_train, y_train

# Train a Random Forest Classifier for task priority
priority_clf = RandomForestClassifier(random_state=42)
priority_clf.fit(X_resampled, y_resampled)

# Evaluate the model
y_pred = priority_clf.predict(X_test)
print(f"Task Priority Model Accuracy: {accuracy_score(y_test, y_pred)}")

# Function to predict priority for new tasks
def predict_priority(task):
    priority = priority_clf.predict(task)
    return label_encoder.inverse_transform(priority)[0]

# Sample data: Employee ID, Skill Set, Current Workload, Performance Score
employee_data = {
    'Employee ID': [101, 102, 103, 104, 105],
    'Skill Set': [['Python', 'ML'], ['Java', 'DB'], ['Python', 'AI'], ['JS', 'React'], ['Python', 'Data Analysis']],
    'Current Workload': [3, 5, 2, 4, 1],
    'Performance Score': [8, 7, 9, 6, 10]
}

# Convert employee data to DataFrame
employee_df = pd.DataFrame(employee_data)

# Function to recommend employee for a task
def recommend_employee(task_skills, employees_df):
    best_employee = None
    best_score = -1
    
    for index, employee in employees_df.iterrows():
        skill_match = len(set(task_skills).intersection(set(employee['Skill Set'])))
        workload_score = 10 - employee['Current Workload']
        performance_score = employee['Performance Score']
        
        total_score = skill_match + workload_score + performance_score
        
        if total_score > best_score:
            best_score = total_score
            best_employee = employee['Employee ID']
    
    return best_employee

# Function to determine complexity based on tech stack
def determine_complexity(tech_stack):
    if "AI" in tech_stack or "Machine Learning" in tech_stack:
        return 9
    elif "React" in tech_stack or "Java" in tech_stack:
        return 7
    elif "Python" in tech_stack or "Data Analysis" in tech_stack:
        return 5
    else:
        return 3

# Function to create a new task and update the task database
def create_task(deadline, dependencies, importance, tech_stack):
    task_id = str(uuid.uuid4())  # Generate random Task ID
    complexity = determine_complexity(tech_stack)  # Determine complexity based on tech stack
    new_task = {
        'Task ID': task_id,
        'Deadline': deadline,
        'Complexity': complexity,
        'Dependencies': dependencies,
        'Importance': importance,
        'Tech Stack': tech_stack,
        'Assigned Employee': None  # Initially, no employee is assigned
    }
    task_database.append(new_task)  # Add the task to the database
    return new_task

# Function to assign a task with AI recommendations
def assign_task_with_ai(task, employees_df):
    # Predict priority
    priority = predict_priority(pd.DataFrame([{
        'Deadline': task['Deadline'],
        'Complexity': task['Complexity'],
        'Dependencies': task['Dependencies'],
        'Importance': task['Importance']
    }]))
    print(f"Task Priority: {priority}")
    
    # Recommend required skills based on tech stack
    print(f"Recommended Skills: {task['Tech Stack']}")
    
    # Recommend employee
    recommended_employee = recommend_employee(task['Tech Stack'], employees_df)
    print(f"Recommended Employee ID: {recommended_employee}")
    
    # Check if the recommended employee already has tasks
    employee_tasks = [t for t in task_database if t['Assigned Employee'] == recommended_employee]
    if employee_tasks:
        print(f"Employee {recommended_employee} is currently assigned to the following tasks:")
        for t in task_database:
            task_df = pd.DataFrame([{
                'Deadline': t['Deadline'],
                'Complexity': t['Complexity'],
                'Dependencies': t['Dependencies'],
                'Importance': t['Importance']
            }])
            priority = predict_priority(task_df)
            print(f"Task ID: {t['Task ID']}, Priority: {priority}")
        
        # Ask the manager if they want to reallocate a previous task or assign the new task
        choice = input("Do you want to reallocate a previous task (R) or assign the new task (N)? ").strip().upper()
        if choice == 'R':
            # Reallocate a previous task
            task_id_to_reallocate = input("Enter the Task ID to reallocate: ").strip()
            task_to_reallocate = next((t for t in task_database if t['Task ID'] == task_id_to_reallocate), None)
            if task_to_reallocate:
                # Assign the new task to the recommended employee
                task['Assigned Employee'] = recommended_employee
                employee_df.loc[employee_df['Employee ID'] == recommended_employee, 'Current Workload'] += 1
                print(f"Task {task['Task ID']} assigned to Employee {recommended_employee}.")
                
                # Reallocate the previous task
                new_employee = input(f"Enter the new Employee ID for Task {task_to_reallocate['Task ID']}: ").strip()
                task_to_reallocate['Assigned Employee'] = int(new_employee)
                employee_df.loc[employee_df['Employee ID'] == int(new_employee), 'Current Workload'] += 1
                employee_df.loc[employee_df['Employee ID'] == recommended_employee, 'Current Workload'] -= 1
                print(f"Task {task_to_reallocate['Task ID']} reallocated to Employee {new_employee}.")
            else:
                print("Invalid Task ID. No reallocation performed.")
        elif choice == 'N':
            # Assign the new task to the recommended employee
            task['Assigned Employee'] = recommended_employee
            employee_df.loc[employee_df['Employee ID'] == recommended_employee, 'Current Workload'] += 1
            print(f"Task {task['Task ID']} assigned to Employee {recommended_employee}.")
        else:
            print("Invalid choice. No action taken.")
    else:
        # Assign the new task to the recommended employee
        task['Assigned Employee'] = recommended_employee
        employee_df.loc[employee_df['Employee ID'] == recommended_employee, 'Current Workload'] += 1
        print(f"Task {task['Task ID']} assigned to Employee {recommended_employee}.")

# Function to display all tasks in the database
def display_tasks():
    if not task_database:
        print("No tasks have been created yet.")
    else:
        print("\n--- Task Database ---")
        for task in task_database:
            print(f"Task ID: {task['Task ID']}")
            print(f"Deadline: {task['Deadline']} days")
            print(f"Complexity: {task['Complexity']}")
            print(f"Dependencies: {task['Dependencies']}")
            print(f"Importance: {task['Importance']}")
            print(f"Tech Stack: {', '.join(task['Tech Stack'])}")
            print(f"Assigned Employee: {task['Assigned Employee']}")
            print("-----------------------------")

# Command-line interface (CLI) for task management
def main_menu():
    print("\n--- Task Management System ---")
    print("1. Create Task")
    print("2. Assign Task with AI Recommendations")
    print("3. Display All Tasks")
    print("4. Exit")

def create_task_menu():
    deadline = int(input("Enter Deadline (days): "))
    dependencies = int(input("Enter Dependencies: "))
    importance = int(input("Enter Importance (1-10): "))
    tech_stack = input("Enter required tech stack (comma-separated): ").split(',')
    tech_stack = [skill.strip() for skill in tech_stack]
    return create_task(deadline, dependencies, importance, tech_stack)

def assign_task_menu():
    if not task_database:
        print("No tasks available to assign. Please create a task first.")
    else:
        print("\nAvailable Tasks:")
        for i, task in enumerate(task_database):
            if task['Assigned Employee'] is None:
                print(f"{i + 1}. Task ID: {task['Task ID']}")
        task_index = int(input("Select a task to assign (enter task number): ")) - 1
        if 0 <= task_index < len(task_database) and task_database[task_index]['Assigned Employee'] is None:
            assign_task_with_ai(task_database[task_index], employee_df)
        else:
            print("Invalid task selection.")

def main():
    while True:
        main_menu()
        choice = input("Enter your choice: ")
        if choice == '1':
            create_task_menu()
        elif choice == '2':
            assign_task_menu()
        elif choice == '3':
            display_tasks()
        elif choice == '4':
            print("Exiting the system. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

# Run the program
if __name__ == "__main__":
    main()
