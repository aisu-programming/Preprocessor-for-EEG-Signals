import optuna
import plotly.io as pio
pio.kaleido.scope.default_width  = 1200
pio.kaleido.scope.default_height = 1200

MODEL = ["EEGNet", "GRU", "LSTM", "ATCNet"][2]
DATASET = ["BcicIv2a", "PhysionetMI", "Ofner"][0]
FRAMEWORK = ["pt", "tf"][0]
STUDY_NAME = f"{MODEL}_{DATASET}_{FRAMEWORK}"

study = optuna.create_study(study_name=STUDY_NAME,
                            storage=f"sqlite:///hps_c_{STUDY_NAME}.db",
                            load_if_exists=True)

fig = optuna.visualization.plot_contour(study, target=lambda ft: ft.values[0], target_name="Valid Acc")
fig.write_image(f"histories_cls_search/{STUDY_NAME}/contour.png")