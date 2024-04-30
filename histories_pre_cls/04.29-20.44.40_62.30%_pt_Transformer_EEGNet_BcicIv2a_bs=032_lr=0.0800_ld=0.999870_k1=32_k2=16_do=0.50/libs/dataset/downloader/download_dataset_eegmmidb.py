import dotenv
dotenv.load_dotenv(".env")
import os
import mne
mne.set_config("MNE_DATA", os.environ["DATASET_DIR"])
mne.set_config("MNE_DATASETS_EEGBCI_PATH", os.environ["DATASET_DIR"])

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import warnings
warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", ModuleNotFoundError)

from moabb.datasets import PhysionetMI
from moabb.paradigms import MotorImagery

try:
    MotorImagery().get_data(dataset=PhysionetMI())
except ValueError:
    print(f"Successfully downloaded all data.")