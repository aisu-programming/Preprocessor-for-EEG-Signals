import optuna
import optuna.visualization
import itertools
import plotly.io as pio
from tqdm import tqdm


PREPROCESSOR = None
# PREPROCESSOR = ["LSTM", "Transformer"][0]
CLASSIFIER = ["EEGNet", "ATCNet", "GRU", "LSTM"][3]
DATASET = ["BcicIv2a", "PhysionetMI", "Ofner"][0]
FRAMEWORK = "pt"  # ["pt", "tf"][0]



if PREPROCESSOR is not None:
    study_name = f"{PREPROCESSOR}_{CLASSIFIER}_{DATASET}_{FRAMEWORK}"
    storage = f"sqlite:///hps_p_{study_name}.db"
    root = "histories_pre"
else:
    study_name = f"{CLASSIFIER}_{DATASET}_{FRAMEWORK}"
    storage = f"sqlite:///hps_c_{study_name}.db"
    root = "histories_cls"

study = optuna.create_study(study_name=study_name,
                            storage=storage,
                            load_if_exists=True)

pio.kaleido.scope.default_width  = 1500
pio.kaleido.scope.default_height = 1500
fig = optuna.visualization.plot_contour(
    study, target=lambda ft: ft.values[0], target_name="Valid Acc")
fig.layout.title = study_name
fig.write_image(f"{root}/{study_name}/contour.png")

# plt.tight_layout()
# plt.savefig(f"{root}/{study_name}/contour_test.png")

params = study.trials[0].params.keys()
for param_1, param_2 in tqdm(list(itertools.product(params, params))):
    if param_1 == param_2: continue
    pio.kaleido.scope.default_width  = 600
    pio.kaleido.scope.default_height = 600
    fig = optuna.visualization.plot_contour(
        study, params=[param_1, param_2], 
        target=lambda ft: ft.values[0], target_name="Valid Acc")
    param_1 = param_1.replace('_', '-')
    param_2 = param_2.replace('_', '-')
    fig.layout.title = f"{study_name}:\n{param_1} / {param_2}"
    fig.write_image(f"{root}/{study_name}/contour_{param_1}_{param_2}.png")