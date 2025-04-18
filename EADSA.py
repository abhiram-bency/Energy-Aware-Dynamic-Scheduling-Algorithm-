import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import random
import time
from collections import deque
import matplotlib.pyplot as plt

class WorkloadPredictor:
    """
    Workload Prediction Module using Machine Learning
    """
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.history = deque(maxlen=1000)  # Stores historical workload data
        self.features = ['cpu_usage', 'task_count', 'task_type', 'execution_time']
        self.is_trained = False
        
    def generate_training_data(self, num_samples=1000):
        """Generate synthetic training data"""
        data = []
        for _ in range(num_samples):
            cpu_usage = random.uniform(0.1, 0.9)
            task_count = random.randint(1, 20)
            task_type = random.choice([0, 1, 2])  # 0: CPU-bound, 1: I/O-bound, 2: Mixed
            execution_time = random.uniform(0.1, 5.0)
            future_workload = cpu_usage * 0.8 + task_count * 0.05 + random.uniform(-0.1, 0.1)
            data.append([cpu_usage, task_count, task_type, execution_time, future_workload])
        
        df = pd.DataFrame(data, columns=self.features + ['future_workload'])
        return df
    
    def train_model(self):
        """Train the workload prediction model"""
        data = self.generate_training_data()
        X = data[self.features]
        y = data['future_workload']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        print(f"Model trained with MAE: {mae:.4f}")
        self.is_trained = True
        
    def predict_workload(self, current_state):
        """Predict future workload based on current system state"""
        if not self.is_trained:
            self.train_model()
            
        # Record current state in history
        self.history.append(current_state)
        
        # Prepare input for prediction
        input_data = pd.DataFrame([current_state], columns=self.features)
        predicted_load = self.model.predict(input_data)[0]
        
        # Clip between 0 and 1
        predicted_load = max(0.0, min(1.0, predicted_load))
        
        return predicted_load

class DVFSController:
    """
    Dynamic Voltage and Frequency Scaling Module
    """
    def __init__(self):
        self.available_frequencies = [0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.2, 2.5]  # GHz
        self.current_frequency = 1.0  # Start with base frequency
        self.energy_consumption = 0
        self.power_profile = {f: f**3 for f in self.available_frequencies}  # Simplified power model (P ∝ f^3)
        
    def adjust_frequency(self, predicted_load):
        """Adjust CPU frequency based on predicted workload"""
        # Simple heuristic: map load to frequency
        if predicted_load < 0.3:
            new_freq = self.available_frequencies[0]  # Lowest frequency
        elif predicted_load < 0.5:
            new_freq = self.available_frequencies[1]
        elif predicted_load < 0.7:
            new_freq = self.available_frequencies[3]
        else:
            new_freq = self.available_frequencies[-1]  # Highest frequency
            
        self.current_frequency = new_freq
        return new_freq
    
    def calculate_energy(self, execution_time):
        """Calculate energy consumption for a given execution time"""
        power = self.power_profile[self.current_frequency]
        energy = power * execution_time
        self.energy_consumption += energy
        return energy

class Task:
    """Task class representing a process to be scheduled"""
    def __init__(self, task_id, task_type, burst_time, priority=None):
        self.task_id = task_id
        self.task_type = task_type  # 0: CPU-bound, 1: I/O-bound, 2: Mixed
        self.burst_time = burst_time
        self.remaining_time = burst_time
        self.priority = priority if priority is not None else random.randint(1, 5)
        self.arrival_time = time.time()
        self.energy_estimate = burst_time * (1 + 0.2 * task_type)  # Simplified energy estimate
        
    def execute(self, time_slice, frequency):
        """Execute the task for a given time slice"""
        actual_time = time_slice * (1.0 / frequency)  # Adjust for frequency scaling
        if self.remaining_time <= actual_time:
            actual_time = self.remaining_time
            self.remaining_time = 0
        else:
            self.remaining_time -= actual_time
            
        return actual_time

class AdaptiveTaskScheduler:
    """
    Adaptive Task Prioritization Module
    """
    def __init__(self):
        self.ready_queue = []
        self.workload_predictor = WorkloadPredictor()
        self.dvfs_controller = DVFSController()
        self.energy_threshold = 100  # Arbitrary energy threshold
        self.total_energy = 0
        self.total_tasks_completed = 0
        self.total_response_time = 0
        self.current_time = 0
        
    def add_task(self, task):
        """Add a new task to the scheduler"""
        self.ready_queue.append(task)
        
    def prioritize_tasks(self, energy_mode=False):
        """Re-order tasks based on current priorities and energy mode"""
        if energy_mode:
            # In energy-saving mode, prioritize tasks that consume less energy
            self.ready_queue.sort(key=lambda x: (x.energy_estimate, x.priority))
        else:
            # Normal mode: prioritize by priority and burst time
            self.ready_queue.sort(key=lambda x: (-x.priority, x.remaining_time))
            
    def schedule(self):
        """Main scheduling loop"""
        print("Starting EADSA scheduler...")
        
        while self.ready_queue:
            # Get current system state for prediction
            current_state = {
                'cpu_usage': len(self.ready_queue) / 20,  # Normalized
                'task_count': len(self.ready_queue),
                'task_type': np.mean([t.task_type for t in self.ready_queue]),
                'execution_time': np.mean([t.remaining_time for t in self.ready_queue])
            }
            
            # Predict workload and adjust frequency
            predicted_load = self.workload_predictor.predict_workload(current_state)
            current_freq = self.dvfs_controller.adjust_frequency(predicted_load)
            
            # Determine if we should enter energy-saving mode
            energy_mode = self.total_energy > self.energy_threshold
            
            # Re-prioritize tasks based on current mode
            self.prioritize_tasks(energy_mode)
            
            # Get the next task to execute
            current_task = self.ready_queue.pop(0)
            
            # Determine time slice based on frequency and priority
            base_time_slice = 0.5  # Default time quantum
            adjusted_time_slice = base_time_slice * (current_task.priority / 5)  # Higher priority gets more time
            
            # Execute the task
            start_time = self.current_time
            executed_time = current_task.execute(adjusted_time_slice, current_freq)
            energy_used = self.dvfs_controller.calculate_energy(executed_time)
            
            # Update statistics
            self.total_energy += energy_used
            self.current_time += executed_time
            
            if current_task.remaining_time > 0:
                # Task not finished, put it back in the queue
                self.ready_queue.append(current_task)
            else:
                # Task completed
                self.total_tasks_completed += 1
                response_time = self.current_time - current_task.arrival_time
                self.total_response_time += response_time
                print(f"Task {current_task.task_id} completed. Energy used: {energy_used:.2f} J")
            
            # Print current status
            print(f"Tasks in queue: {len(self.ready_queue)}, Current freq: {current_freq} GHz, "
                  f"Predicted load: {predicted_load:.2f}, Energy mode: {energy_mode}")
            
            # Simulate some delay between tasks
            time.sleep(0.1)
            
        # Print summary statistics
        avg_response_time = self.total_response_time / self.total_tasks_completed if self.total_tasks_completed > 0 else 0
        print("\nScheduling completed!")
        print(f"Total energy consumed: {self.total_energy:.2f} J")
        print(f"Total tasks completed: {self.total_tasks_completed}")
        print(f"Average response time: {avg_response_time:.2f} s")

def simulate_workload(scheduler, num_tasks=20):
    """Generate a simulated workload for testing"""
    task_types = [0, 1, 2]  # CPU-bound, I/O-bound, Mixed
    for i in range(num_tasks):
        task_type = random.choice(task_types)
        burst_time = random.uniform(1.0, 10.0)
        priority = random.randint(1, 5)
        task = Task(i, task_type, burst_time, priority)
        scheduler.add_task(task)
        print(f"Added Task {i}: Type {task_type}, Burst {burst_time:.2f}s, Priority {priority}")
        time.sleep(random.uniform(0.1, 0.5))  # Simulate random arrival times

def main():
    """Main function to run the EADSA simulation"""
    scheduler = AdaptiveTaskScheduler()
    
    # Start workload simulation in a separate thread
    import threading
    workload_thread = threading.Thread(target=simulate_workload, args=(scheduler, 15))
    workload_thread.start()
    
    # Start scheduling
    scheduler.schedule()
    workload_thread.join()

if __name__ == "__main__":
    main()
