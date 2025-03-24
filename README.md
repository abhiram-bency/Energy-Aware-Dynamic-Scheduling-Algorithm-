# Energy-Aware-Dynamic-Scheduling-Algorithm-
Energy-Aware Dynamic Scheduling Algorithm -Energy-Efficient CPU Scheduling Algorithm
# Energy-Aware Dynamic Scheduling Algorithm (EADSA) - C++ Implementation  

## 📌 Overview  

The **Energy-Aware Dynamic Scheduling Algorithm (EADSA)** is a CPU scheduling approach designed to optimize both **energy efficiency** and **computational performance**. Unlike traditional scheduling methods that focus only on execution speed, EADSA integrates:  
- **Workload Prediction Module**: Forecasts future CPU workload using **linear regression estimation**.  
- **Dynamic Voltage and Frequency Scaling (DVFS)**: Adjusts **CPU frequency** based on workload demand.  
- **Adaptive Task Prioritization**: Reorders tasks based on **priority and energy constraints** to maximize efficiency.  

This implementation **simulates** a CPU scheduling environment, dynamically adjusting power usage while ensuring efficient task execution.  

---

## ⚙️ Features  

✔ **Workload Prediction**: Estimates future CPU workload using historical data.  
✔ **DVFS Optimization**: Dynamically scales CPU frequency to **save power**.  
✔ **Task Prioritization**: Sorts tasks based on priority and energy requirements.  
✔ **Simulation Mode**: Runs a test scenario demonstrating EADSA’s efficiency.  

---

## 🛠️ Installation & Compilation  

### **1️⃣ Clone the Repository**  
```sh
git clone https://github.com/yourusername/EADSA_CPP.git
cd EADSA_CPP
