# Every Character Counts: From Vulnerability to Defense in Phishing Detection

This repository contains the implementation and dataset for the paper "Every Character Counts: From Vulnerability to Defense in Phishing Detection"

## About 

This repository hosts the custom-built email dataset, which combined data from multiple sources for our neural phishing detection, and the implementaion described in the original paper. We add instructions and full code in order to allow other researchers to easily replicate our research or validate our approach. 

## Instructions

### Installing Pre-requisites

In order to run the code you will need to install the requirements found in requirements.txt.

```bash
pip install -r requirements.txt
```

Paper reproduction:
1. Go to 'charbilstm', 'chargru', 'charcnn' folder and run the following scripts:  

```bash
python standard_training.py` 

python adversarial_training.py`  

python adversarial_testing_and_training.py.
```

2. After training, a .h5 model file will be saved in the same folder. Run the explainability script to generate Grad-CAM visualizations:

```bash
python explain_char*.py --model_path your_model_file.h5
```

Replace explain_char*.py with:
    - explain_charcnn.py
    - explain_chargru.py
    - explain_charbilstm.py

3. To compare CharGRU (adversarial training) with LLama 3.2 run:

```bash
python adversarial_chargru_subset_eval.py"
```

```bash
python llama3.2_subset_eval.py"
```
