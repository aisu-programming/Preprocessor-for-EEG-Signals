cd datasets
wget https://physionet.org/static/published-projects/eegmmidb/eeg-motor-movementimagery-dataset-1.0.0.zip
unzip eeg-motor-movementimagery-dataset-1.0.0.zip
rm eeg-motor-movementimagery-dataset-1.0.0.zip
mv files MNE-eegbci-data/files/eegmmidb/1.0.0  # Hasn't test