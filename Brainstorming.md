## EEG Problems
1. Problem: Low SNR (Signal-to-Noise Ratio)
   - Solution: Data Cleaning
   - Solution: Data Super-Resolution
   - Solution: Feature Extraction / Selection

1. Problem: Few Availabe Datasets
   - Solution: Data Augmentation

1. Problem: Complicated Data
   - Solution: Deep Learning Methods
     - Option: CNNs
     - Option: RNN
     - Option: LSTM
     - Option: Transformers
     - Option: Geometric Learning (~~Euclidean~~ Riemannian)

## EEG Project Ideas
1. EEG GAN / Diffusion
   - Reference:
     - [EEG Signal Reconstruction Using a Generative Adversarial Network With Wasserstein Distance and Temporal-Spatial-Frequency Loss (Year: 2020 / Cited: 47)](https://www.frontiersin.org/articles/10.3389/fninf.2020.00015/full)
     - [EEG-GAN: Generative adversarial networks for electroencephalograhic (EEG) brain signals (Year: 2018 / Cited: 277)](https://arxiv.org/abs/1806.01875)
     - [Deep EEG super-resolution: Upsampling EEG spatial resolution with Generative Adversarial Networks (Year: 2018 / Cited: 49)](https://www.semanticscholar.org/paper/Deep-EEG-super-resolution%3A-Upsampling-EEG-spatial-Corley-Huang/be380a48c62308414da2706d289b6d526df19f7c)
     - [Super-Resolution for Improving EEG Spatial Resolution using Deep Convolutional Neural Network—Feasibility Study (Year: 2019 / Cited: 27)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6928936/)
   - Goal
     Propose a GAN-style model to either clean (denoise), augment, or super-resolution the data.
     Super-resolution can channel-wise or timestep-wise.

1. (Unclear) Data Augmentation
   - Goal
     Propose a (novel) method/pipeline to improve the ability of the model for eliminating unrelated signals.
     - Mixup all datasets with different subjects and tasks
     - Do Noise Injection / Jittering on the data

## EEG Dataset Options

1.  BCI Competition IV 2a: Motor Imagery
   Signal Type: Oscillatory
   Electrodes: 22
   Subjects: 9
   Sessions: 2
   Total trails: 288 x 9 x 2 = 5184
   Interval: 4 secs
   Sample Rate: 250 Hz
   Classes: 4
   Class Balance: No
   Arrangement: ?
   Used by:
   - [ShallowConvNet](https://arxiv.org/abs/1703.05051)
   - [EEGNet](https://arxiv.org/abs/1611.08024)
   - [MAtt](https://arxiv.org/abs/2210.01986)

1. P300 Event-Related Potential (P300)
   Type: ERP
   Electrodes: 64
   Subjects: 18 -> 15 (Filtered in Original Dataset or EEGNet?)
   Sessions: ?
   Total trails: ~2000 x 15 = ~30000
   Interval: 1 secs
   Sample Rate: 512 Hz
   Classes: 2
   Class Balance: Yes (~5.6 : 1)
   Arrangement: 10-10
   Used by:
   - [EEGNet](https://arxiv.org/abs/1611.08024)

1. Feedback Error-Related Negativity (ERN)
   Type: ERP
   Electrodes: 56
   Subjects: 26
   Sessions: ?
   Total trails: 340 x 26 = 8840
   Interval: 1.25 secs
   Sample Rate: 600 Hz
   Classes: 2
   Class Balance: Yes (~3.4 : 1)
   Arrangement: 10-20
   Used by:
   - [EEGNet](https://arxiv.org/abs/1611.08024)

1. Movement-Related Cortical Potential (MRCP)
   Type: ERP + Oscillatory
   Electrodes: 256
   Subjects: 13
   Sessions: ?
   Total trails: ~1100 x 13 = ~14300
   Interval: 1.5 secs
   Sample Rate: 1024 Hz
   Classes: 2
   Class Balance: No
   Arrangement: ?
   Used by:
   - [EEGNet](https://arxiv.org/abs/1611.08024)

1. PhysioNet: Motor Execution + Moter Imagery
   Type: Oscillatory?
   Electrodes: 64
   Subjects: 109
   Sessions: ?
   Total trails: ?
   Interval: ?
   Sample Rate: 160 Hz
   Classes: 1+4+4 = 9 (Baseline: 1 / Execution: 4 / Imagery: 4)
   Class Balance: ?
   Arrangement: ?
   Used by:
   - [GRUGate Tranformer](https://ieeexplore.ieee.org/document/9630210)

<!-- 1. ...
   Type: 
   Electrodes: 
   Subjects: 
   Sessions: 
   Total trails: 
   Interval: 
   Sample Rate: 
   Classes: 
   Class Balance: 
   Arrangement: 
   Used by:
   - ... -->

1. [PhysioNet: Auditory evoked potential EEG-Biometric dataset](https://physionet.org/content/auditory-eeg/1.0.0/)
   Electrodes: 4


## EEG Dataset Tools
1. :::spoiler [BioSig](https://sourceforge.net/p/biosig/wiki/Home/)
   - Language: Primarily C, with interfaces for MATLAB/Octave
   - Support signal types: EEG, ECoG, EMG, ECG, HRV, and more
   - Support formats: EDF, BDF, GDF, and over 50 other data formats
   - Platforms: Windows, Linux, and macOS
   - Data processing functions:
     Filtering, Artifact Removal, Feature Extraction, Signal Classification...
   - Others:
     Free and Open Source, provides a comprehensive toolkit for the analysis and management of biomedical signals.
   
1. :::spoiler [EEGLAB](https://sccn.ucsd.edu/eeglab/downloadtoolbox.php)
   - Language: MATLAB
   - Support signal types: Primarily **EEG**
   - Support formats: Supports various EEG data formats through plugins
   - Platforms: Windows, Linux, and macOS (MATLAB environment)
   - Data processing functions:
     Data Import, Preprocessing, **Visualization**, Time-Frequency Analysis, Statistical Analysis...
   - Others: Free and Open Source, Extensive **GUI**, Large collection of plugins
   
1. :::spoiler FieldTrip
   - Language: MATLAB
   - Support signal types: **EEG**, MEG, iEEG
   - Support formats: Supports a wide range of electrophysiological data formats
   - Platforms: Windows, Linux, and macOS (MATLAB environment)
   - Data processing functions:
     Preprocessing, Time-Frequency Analysis, Source Reconstruction, Statistical Testing...
   - Others:
     Free and Open Source, Detailed documentation and tutorials for advanced analyses

1. :::spoiler Brainstorm
   - Language: MATLAB
   - Support signal types: **EEG**, MEG, iEEG
   - Support formats: A wide range of data formats supported
   - Platforms: Windows, Linux, and macOS (MATLAB environment)
   - Data processing functions:
     **Visualization**, Analysis Pipeline, Statistical Analysis, Source Modeling...
   - Others:
     Free and Open Source, User-friendly **GUI**,
     Suitable for educational purposes and researchers without programming skills

1. :::spoiler MNE-Python
   - Language: **Python**
   - Support signal types: **EEG**, MEG
   - Support formats: FIFF (native format), other formats through conversion
   - Platforms: Windows, Linux, and macOS
   - Data processing functions:
     Data Preprocessing, **Visualization**, Decoding, Source Localization, Statistical Analysis...
   - Others:
     Free and Open Source, Integrates well with the Python scientific computing ecosystem
     
1. :::spoiler NeuroKit2
   - Language: **Python**
   - Support signal types: **EEG**, ECG, PPG, EMG, and more
   - Support formats:
     Compatible with data from various sources and formats through Python
   - Platforms: Windows, Linux, and macOS
   - Data processing functions:
     Signal Processing, Feature Extraction, **Visualization**, Analysis...
   - Others:
     Free and Open Source, focuses on ease of use for psychological and physiological research

1. :::spoiler PyEEG
   - Language: **Python**
   - Support signal types: Primarily **EEG**
   - Support formats: Works with numerical arrays in Python, making it flexible with data formats
   - Platforms: Windows, Linux, and macOS
   - Data processing functions: Feature Extraction from EEG signals...
   - Others:
     Free and Open Source, provides basic functions for EEG processing, suitable for research

1. :::spoiler BrainFlow
   - Language: **Python** (with support for other languages)
   - Support signal types: **EEG**, EMG, ECG, and more
   - Support formats: Compatible with a wide range of biosignal acquisition devices
   - Platforms: Windows, Linux, and macOS
   - Data processing functions: Data Acquisition, Signal Processing, Real-time Analysis...
   - Others: Free and Open Source, offers a unified API for different biosignal boards

1. :::spoiler PyCaret
   - Language: **Python**
   - Support signal types: General-purpose for machine learning, applicable to biosignal data
   - Support formats:
     Compatible with any data that can be transformed into a pandas DataFrame
   - Platforms: Windows, Linux, and macOS
   - Data processing functions:
     Automated Machine Learning Workflow, including preprocessing, feature engineering, model tuning...
   - Others:
     Free and Open Source, simplifies machine learning tasks, making it easier to apply complex models to biosignal analysis