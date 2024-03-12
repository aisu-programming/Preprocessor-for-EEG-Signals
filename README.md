# Signal-Preprocessor-for-EEG

## Resource
- [BrainStorming](https://hackmd.io/Z5uL78LPQxOmcMfXGpx-yg)
- [Reading List](https://hackmd.io/I66tk7x0QZSzLT109ARyyA)
- [Slide](https://docs.google.com/presentation/d/1a-_5RynrPjn3GtYHO_E8XeGg9G7NTmGgocn0C2NYVrA/edit?usp=sharing)

## Tutorial
1. Clone this project by: <br>
   ```sh
   $ git clone --recurse-submodules https://github.com/aisu-programming/Preprocessor-for-EEG-Signal.git
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
  $ pip install -r requirements.txt
  ```
- Can also use conda. <br>
- ~~Must use Python <= 3.9 because of tensorflow.~~

> [!TIP]
> When running MacOS on Apple Silicon, also install `tensorflow-metal` to utilize GPU usage:
> ```sh
> $ pip install tensorflow-metal
> ```
> You'll want to run with a higher batch size to get the benefits.

## Example
Here's how you can run _baselines.py_ from your shell:
 - first, set __TF_USE_LEGACY_KERAS__ to True.
   ```sh
   $ export TF_USE_LEGACY_KERAS=True
   ```
 - then, run the script.
   ```sh
   $ python3 baselines.py
   ```