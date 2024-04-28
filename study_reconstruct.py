import os
import optuna
import datetime


MODEL = "GRU"
DATASET = "BcicIv2a"
STUDY_NAME = f"{MODEL}_{DATASET}_pt"
STORAGE = f"sqlite:///hps_c_{STUDY_NAME}_reconstruction.db"

study = optuna.create_study(study_name=STUDY_NAME, storage=STORAGE, load_if_exists=True)
print(len(study.trials))

root = f"histories_cls_search/{STUDY_NAME}"
for idx, dir in enumerate(os.listdir(root)):
    if not os.path.isdir(f"{root}/{dir}"): continue
    args = {}
    with open(f"{root}/{dir}/args.txt") as args_file:
        line_list = args_file.readlines()
        for line in line_list:
            line = line.replace('\n', '')
            key, value = line.split('=', 1)
            if key in [ "model",
                        "dataset",
                        "framework",
                        "study_name",
                        "mixed_signal",
                        "mixing_source",
                        "mixing_duplicate",
                        "epochs",
                        "device",
                        "num_workers",
                        "save_plot",
                        "show_plot",
                        "auto_hps",
                        "save_dir" ]:
                continue
            try:
                value = float(value)
                if value.is_integer():
                    value = int(value)
            except:
                pass
            args[key] = value
    if "num_layers" not in args: args["num_layers"] = 2
    if args["num_layers"] > 4: continue
    if args["hid_channels"] > 128: continue

    distributions = {
        'num_layers': optuna.distributions.CategoricalDistribution(choices=(1, 2, 3, 4)),
        'hid_channels': optuna.distributions.CategoricalDistribution(choices=(16, 32, 64, 128)),
        'batch_size': optuna.distributions.CategoricalDistribution(choices=(8, 16, 32, 64)),
        'learning_rate': optuna.distributions.FloatDistribution(high=0.05, log=True, low=0.0005, step=None),
        'lr_decay': optuna.distributions.FloatDistribution(high=0.99995, log=False, low=0.99985, step=None)
    }
    study.add_trial(optuna.trial.FrozenTrial(
        number=idx,
        state=optuna.trial.TrialState.COMPLETE,
        value=None,
        values=[ float(dir.split('%')[0])/100 ],
        params=args,
        trial_id=idx+1,
        datetime_start=datetime.datetime(2024, 4, 26, 12, 34, 56, 789100),
        datetime_complete=datetime.datetime(2024, 4, 26, 12, 34, 56, 789100),
        distributions=distributions,
        user_attrs={},
        system_attrs={},
        intermediate_values={},
    ))

print(len(study.trials))