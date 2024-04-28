import os
import optuna

MODEL = "LSTM"
DATASET = "BcicIv2a"
STUDY_NAME = f"{MODEL}_{DATASET}_pt"
STORAGE = f"sqlite:///hps_c_{STUDY_NAME}.db"
# STORAGE = f"sqlite:///hps_c_{STUDY_NAME}_reconstruction.db"

study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE)
for trial in study.trials:
    if trial.params["num_layers"] != 2:
        print(trial)
print(len(study.trials), len(os.listdir(f"histories_cls_search/{STUDY_NAME}")))