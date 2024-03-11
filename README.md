# Signal-Preprocessor-for-EEG

## Resource
- [BrainStorming](https://hackmd.io/Z5uL78LPQxOmcMfXGpx-yg)
- [Reading List](https://hackmd.io/I66tk7x0QZSzLT109ARyyA)
- [Slide](https://docs.google.com/presentation/d/1a-_5RynrPjn3GtYHO_E8XeGg9G7NTmGgocn0C2NYVrA/edit?usp=sharing)

## Tutorial
1. Clone this project by: `git clone --recurse-submodules https://github.com/aisu-programming/Preprocessor-for-EEG-Signal.git`.
2. Download the above datasets and put them in directory "_datasets_".
3. Rename the file "_.env_sample_" to "_.env_".

## Dataset
1. BCI Competition IV 2a <br>
   [Train subset with labels + Test subset without labels + Documents](https://www.bbci.de/competition/download/competition_iv/BCICIV_2a_gdf.zip) <br>
   [Test subset labels](https://www.bbci.de/competition/iv/results/ds2a/true_labels.zip)

## Environment
With pip: `pip install -r requirements.txt`. <br>
Can also use conda. <br>
Must use Python <= 3.9 because of tensorflow.