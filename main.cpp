#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace std;

// Structure to represent a task
struct Task {
    int id;
    int priority;
    int energyRequirement;
};

// Class for the Energy-Aware Dynamic Scheduling Algorithm (EADSA)
class EADSA {
private:
    double cpuFrequency;      // CPU frequency in GHz
    double energyConsumption; // Energy consumption in arbitrary units
    vector<int> pastWorkloads;

public:
    EADSA() {
        cpuFrequency = 2.0;      // Default CPU frequency (GHz)
        energyConsumption = 50;  // Default energy consumption
    }

    // Function to train workload prediction (Simple Linear Regression Estimation)
    void trainWorkloadModel(vector<int> workloads) {
        pastWorkloads = workloads;
    }

    // Function to predict future workload using a simple linear trend estimation
    double predictWorkload(int timeStep) {
        int n = pastWorkloads.size();
        if (n == 0) return 50; // Default workload if no data is available

        double sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
        for (int i = 0; i < n; i++) {
            sumX += i;
            sumY += pastWorkloads[i];
            sumXY += i * pastWorkloads[i];
            sumXX += i * i;
        }

        double slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
        double intercept = (sumY - slope * sumX) / n;

        return slope * timeStep + intercept;
    }

    // Function to adjust CPU frequency based on predicted workload
    void adjustDVFS(double predictedWorkload) {
        if (predictedWorkload > 70) {
            cpuFrequency = 3.0;  // Increase frequency for high workload
        } else if (predictedWorkload < 30) {
            cpuFrequency = 1.2;  // Decrease frequency for low workload
        } else {
            cpuFrequency = 2.0;  // Keep default
        }

        energyConsumption = 100 / cpuFrequency; // Energy inversely proportional to frequency
    }

    // Function to prioritize tasks based on energy constraints and urgency
    vector<Task> adaptiveTaskPrioritization(vector<Task>& tasks) {
        sort(tasks.begin(), tasks.end(), [](Task& a, Task& b) {
            if (a.priority == b.priority)
                return a.energyRequirement > b.energyRequirement; // Higher energy requirement executes last
            return a.priority > b.priority; // Higher priority first
        });
        return tasks;
    }

    // Function to simulate the scheduling process
    void runSimulation() {
        vector<int> sampleWorkloads = {40, 50, 60, 65, 70, 75, 80, 85, 90, 95};
        trainWorkloadModel(sampleWorkloads);

        vector<Task> tasks = {
            {1, 3, 20},
            {2, 1, 10},
            {3, 2, 15},
            {4, 3, 5}
        };

        for (int t = 10; t < 15; t++) {
            double predictedWorkload = predictWorkload(t);
            adjustDVFS(predictedWorkload);
            vector<Task> sortedTasks = adaptiveTaskPrioritization(tasks);

            cout << "Time: " << t << endl;
            cout << "Predicted Workload: " << predictedWorkload << endl;
            cout << "CPU Frequency Adjusted to: " << cpuFrequency << " GHz" << endl;
            cout << "Energy Consumption: " << energyConsumption << " units" << endl;
            cout << "Task Execution Order: ";
            for (const auto& task : sortedTasks) {
                cout << task.id << " ";
            }
            cout << endl << "----------------------------------------" << endl;
        }
    }
};

// Main function
int main() {
    EADSA scheduler;
    scheduler.runSimulation();
    return 0;
}
