# Preprocessor-for-EEG-Signal

## Resource
- [BrainStorming](https://hackmd.io/Z5uL78LPQxOmcMfXGpx-yg)
- [Reading List](https://hackmd.io/I66tk7x0QZSzLT109ARyyA)
- [Slide](https://docs.google.com/presentation/d/1a-_5RynrPjn3GtYHO_E8XeGg9G7NTmGgocn0C2NYVrA/edit?usp=sharing)

## Tutorial
1. Clone this project.
   - For Windows:
     ```sh
     git clone --recurse-submodules https://github.com/aisu-programming/Preprocessor-for-EEG-Signal.git
     ```
   - For Mac: (Unclear, need to test)
     ```sh
     git clone https://github.com/aisu-programming/Preprocessor-for-EEG-Signal.git
     cd Preprocessor-for-EEG-Signal
     git submodule init
     git submodule update
     ```
2. Download the datasets below and put them in directory "_datasets_".
3. Rename the file "_.env_sample_" to "_.env_".

## Dataset
1. `BCI Competition IV 2a` - *place the contents of these downloads into the same folder with this name.* <br>
   - [Train subset with labels + Test subset without labels + Documents](https://www.bbci.de/competition/download/competition_iv/BCICIV_2a_gdf.zip) <br>
   - [Test subset labels](https://www.bbci.de/competition/iv/results/ds2a/true_labels.zip)

## Environment
- With pip:
  ```sh
  pip install -r requirements.txt
  ```
- Can also use conda. <br>

> [!TIP]
> When running MacOS on Apple Silicon, also install `tensorflow-metal` to utilize GPU usage:
> ```sh
> pip install tensorflow-metal
> ```
> You'll want to run with a higher batch size to get the benefits.

## Example: Baselines
Here's how you can run _baselines_tensorflow.py_ from your shell:
1. If you are working on a Mac platform, set __TF_USE_LEGACY_KERAS__ to True.
   ```sh
   export TF_USE_LEGACY_KERAS=True
   ```
2. Run the script.
   ```sh
   python3 _baselines_tensorflow.py
   ```