import numpy as np
import pickle
import json
import time

import datetime
import config
import utils_data
from sklearn.model_selection import train_test_split, cross_val_score

import xgboost as xgb
import optuna
# config
batch_size = config.batch_size
nb_cl_first = config.nb_cl_first
nb_cl = config.nb_cl
nb_groups = config.nb_groups
nb_total = config.nb_total
epochs = config.epochs
lr = config.lr
wght_decay = config.wght_decay
use_weight_decay_in_exemplar = config.use_weight_decay_in_exemplar
lr_patience = config.lr_patience
stop_patience = config.stop_patience
stop_floor_ep = config.stop_floor_ep
factor = config.factor
min_lr = config.min_lr
nb_cluster = config.nb_cluster

# path
data_path = config.data_path
x_path = config.x_path
y_path = config.y_path
x_path_valid = config.x_path_valid
y_path_valid = config.y_path_valid
save_path = config.save_path

### Initialization of some variables ###
loss_batch = []
files_protoset = []
accuracy_all = []
### Save Results ###
task_accuracy_all = []
iteration_results = []
forgetting_scores = []
task_best_acc_list = []
### Iteration timing and results ###
iteration_timing_results = []

total_classes = nb_cl_first + (nb_groups * nb_cl)  # 22 + (4 * 5) = 42
for _ in range(total_classes):
    files_protoset.append([])

### Random mixing ###
print("Mixing the classes and putting them in batches of classes...")
np.random.seed(config.SEED)


order = np.arange(total_classes)
mixing = np.arange(total_classes)
np.random.shuffle(mixing)


### Preparing the files per group of classes ###
print("Creating a training set ...") 
files_train = utils_data.prepare_files(x_path, y_path, mixing, order, nb_groups, nb_cl, nb_cl_first) 
files_valid = utils_data.prepare_files(x_path_valid, y_path_valid, mixing, order, nb_groups, nb_cl, nb_cl_first) 

### Save the mixing and order ###
with open(f"{nb_cl}mixing.pickle", 'wb') as fp:
    pickle.dump(mixing, fp)

with open(f"{nb_cl}settings_mlp.pickle", 'wb') as fp:
    pickle.dump(order, fp)
    pickle.dump(files_train, fp)

print(datetime.datetime.now())
##### ------------- Main Algorithm START -------------#####
for itera in range(nb_groups + 1):
    iteration_start_time = time.time()
    print(f'Batch of classes number {itera+1} arrives ...')
    
    if itera == 0:
        cur_nb_cl = nb_cl_first
        idx_iter = files_train[itera]
        prev_idx_iter = idx_iter.copy()

    else:
        cur_nb_cl = nb_cl
        idx_iter = files_train[itera][:]
        
        total_cl_now = nb_cl_first + ((itera-1) * nb_cl)
        nb_protos_cl = int(np.ceil(nb_total * 1.0 / total_cl_now))

        for i in range(nb_cl_first + (itera-1)*nb_cl): 
            tmp_var = files_protoset[i]
            selected_exemplars = tmp_var[0:min(len(tmp_var),nb_protos_cl)]
            idx_iter += selected_exemplars
        idx_iter = np.concatenate((prev_idx_iter, idx_iter))



    print(f'Task {itera + 1}: Training {cur_nb_cl} classes...') 

    X_full, y_full = utils_data.read_data(x_path, y_path, mixing, idx_iter)
    X_val, y_val = utils_data.read_data(x_path_valid, y_path_valid, mixing, files_valid[itera])


    X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=42,stratify=y_full)
    # 4. Create and train the XGBoost model
    dtrain = xgb.DMatrix(X_full, label=y_full)
    dtest = xgb.DMatrix(X_test, label=y_test)
    for est in [1000]:
        print("Est------------------")
        print(est)
        model = xgb.XGBClassifier(
            num_class=nb_cl, 
            eval_metric='mlogloss',
            use_label_encoder=False,
            max_depth=4,
            learning_rate=0.3,
            n_estimators=est,
            random_state=42,
            device='cpu',
            n_gpus=0
        )

        scores = cross_val_score(model, X_train, y_train, cv=5)
        print(scores)
        print(datetime.datetime.now())
        
        # Store iteration timing and results
        iteration_time = time.time() - iteration_start_time
        iteration_timing_results.append({
            'iteration': itera,
            'time_seconds': iteration_time,
            'cross_val_scores': scores.tolist(),
            'mean_cv_score': float(scores.mean()),
            'std_cv_score': float(scores.std())
        })

    continue
    def objective(trial):
        # Suggest hyperparameters
        params = {
            'max_depth': trial.suggest_int('max_depth', 2, 10),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        }

        # Create an XGBoost classifier with the suggested hyperparameters
        model = xgb.XGBClassifier(**params, n_estimators=200, objective='multi:softmax', random_state=42)

        # Perform 5-fold cross-validation and return the mean accuracy
        scores = cross_val_score(model, X_full, y_full, cv=5, scoring='accuracy')
        return scores.mean()
    
    # Create an Optuna study with TPE sampler
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())

    # Optimize the study for 100 trials
    study.optimize(objective, n_trials=100)

    # Print the best hyperparameters and best score
    print(f"Best hyperparameters: {study.best_params}")
    print(f"Best score: {study.best_value:.4f}")

# Write iteration timing and results to JSON file
if iteration_timing_results:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    json_filename = f"iteration_results_{timestamp}.json"
    with open(json_filename, 'w') as f:
        json.dump(iteration_timing_results, f, indent=2)
    print(f"\nIteration results saved to {json_filename}")

