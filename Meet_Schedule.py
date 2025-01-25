import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Load the dataset
df = pd.read_csv("team_tasks_dataset.csv")

# Convert time columns to datetime
df["task_start_time"] = pd.to_datetime(df["task_start_time"], errors='coerce')
df["task_end_time"] = pd.to_datetime(df["task_end_time"], errors='coerce')

# Handle missing values
df["task_priority"] = df["task_priority"].fillna("unknown")
df = df.dropna(subset=["task_start_time", "task_end_time"])

# Function to round time to the nearest 30 minutes
def round_to_nearest_30_minutes(dt):
    if dt.minute < 15:
        return dt.replace(minute=0, second=0, microsecond=0)
    elif dt.minute < 45:
        return dt.replace(minute=30, second=0, microsecond=0)
    else:
        return (dt + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)

# Function to find meeting slots
def find_meeting_slots(dataframe, office_start="09:00", office_end="17:00"):
    # Define office hours
    office_start = datetime.strptime(office_start, "%H:%M").time()
    office_end = datetime.strptime(office_end, "%H:%M").time()
    
    # Filter tasks by low priority
    low_priority_df = dataframe[dataframe["task_priority"] == "low"]
    
    if low_priority_df.empty:
        print("No low-priority tasks found.")
        return [("10:00", 0)]

    # Create a list of rounded time slots for each low-priority task
    time_slots = []
    for _, row in low_priority_df.iterrows():
        start = round_to_nearest_30_minutes(row["task_start_time"])
        end = round_to_nearest_30_minutes(row["task_end_time"])
        # Generate 30-minute intervals
        time_slots.extend([start + timedelta(minutes=30) * i for i in range(int((end - start).total_seconds() // 1800))])

    # Count occurrences of each time slot
    time_slot_counts = pd.Series(time_slots).value_counts()

    # Find time slots during office hours
    meeting_slots = {}
    for time, count in time_slot_counts.items():
        if office_start <= time.time() <= office_end:
            meeting_slots[time] = count

    # If no suitable slots are found, suggest a default time
    if not meeting_slots:
        print("No overlapping slots found. Suggesting default meeting time.")
        return [("10:00", 0)]
    
    # Sort by member count and pick top 3 slots
    sorted_slots = sorted(meeting_slots.items(), key=lambda x: x[1], reverse=True)[:3]
    return [(slot[0].strftime("%I:%M %p"), slot[1]) for slot in sorted_slots]

# Find available meeting slots
meeting_slots = find_meeting_slots(df)

# Output top 3 meeting slots
if meeting_slots:
    print("\nTop 3 Suggested Meeting Times Based on Low-Priority Tasks:")
    for time, count in meeting_slots:
        print(f"{time}:{count} members available")
else:
    print("No suitable meeting slots found.")
