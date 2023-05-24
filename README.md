This folder contains the code for the numerical simulations of the paper "Online Inventory Problems: Beyond the i.i.d. Setting with Online Convex Optimization".

## Installing the requirements

You need first to install the requirements. In a command line, you can create a virual environment with the right requirements by running:
```
virtualenv .env
source .env/bin/activate
pip install -r requirements.txt
```

You can now run the notebooks for the simulations with synthetic data:
* `synthetic_experiments/setting_1.ipynb`
* `synthetic_experiments/setting_2.ipynb`
* `synthetic_experiments/setting_3.ipynb`

## Downloading the real-world data

To run the experiments with real-world data you also need to download these by running the script `m5_experiments/m5_data_import_script.py`.
```
python m5_experiments/m5_data_import_script.py
```
Then, you will be able to run the notebooks for the real-world data simulations:
* `m5_experiments/setting_4.ipynb`
* `m5_experiments/setting_5.ipynb`
