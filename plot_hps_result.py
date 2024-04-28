import optuna
import plotly.io as pio
pio.kaleido.scope.default_width  = 1200
pio.kaleido.scope.default_height = 1200


# PREPROCESSOR = None
PREPROCESSOR = ["LSTM", "Transformer"][0]
CLASSIFIER = ["EEGNet", "GRU", "LSTM", "ATCNet"][0]
DATASET = ["BcicIv2a", "PhysionetMI", "Ofner"][0]
FRAMEWORK = "pt"  # ["pt", "tf"][0]

if PREPROCESSOR is not None:
    study_name = f"{PREPROCESSOR}_{CLASSIFIER}_{DATASET}_{FRAMEWORK}"
    storage = f"sqlite:///hps_p_{study_name}.db"
    root = "histories_pre_search"
else:
    study_name = f"{CLASSIFIER}_{DATASET}_{FRAMEWORK}"
    storage = f"sqlite:///hps_c_{study_name}.db"
    root = "histories_cls_search"

study = optuna.create_study(study_name=study_name,
                            storage=storage,
                            load_if_exists=True)

fig = optuna.visualization.plot_contour(
    study, target=lambda ft: ft.values[0], target_name="Valid Acc")
fig.write_image(f"{root}/{study_name}/contour.png")