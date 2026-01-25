import json
import numpy as np
import matplotlib.pyplot as plt
with open('iteration_results_20260116_194542.json','r') as tramel_results_file:
    tramel_results = json.load(tramel_results_file)

with open('xgb_iteration_results_20260118_180220.json','r') as xgb_results_file:
    xgb_results = json.load(xgb_results_file)

xgb_success = [entry['mean_cv_score'] for entry in xgb_results]
xgb_time = [entry['time_seconds'] for entry in xgb_results]

tramel_success = [np.mean(entry['task_accuracies']) for entry in tramel_results]
tramel_time = [entry['time_seconds'] for entry in tramel_results]

plt.plot([i for i in range(len(xgb_success))], xgb_success, label='XGBoost', marker='x')
plt.plot([i for i in range(len(xgb_success))], tramel_success, label='Tramel', marker='o')
plt.xlabel('Task')
plt.ylabel('Accuracy in %')
plt.title('XGBoost vs Tramel PERFORMANCE on CICAndMal 2017')
plt.legend()
plt.show()


plt.plot([i for i in range(len(xgb_time))], xgb_time, label='XGBoost', marker='x')
plt.plot([i for i in range(len(tramel_time))], tramel_time, label='Tramel', marker='o')
plt.title('XGBoost vs Tramel TIME on CICAndMal 2017')
plt.xlabel('Task')
plt.ylabel('Execution time in second')
plt.yscale('log')
plt.ylim(1,)
plt.legend()
plt.show()

