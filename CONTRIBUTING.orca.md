# Contributing

## File Naming

For new files, please add `orca` to the filename. For example, `run_docker.orca.sh` to avoid conflicts with the main repository.

## Pull Request

Ensure you are submitting to the `dev` branch of the current repo. By default, GitHub will point you to the `main` branch of the original `IsaacLab-Arena` repository, which is not right.

## Huggingface

You may need to download datasets, models, or files from huggingface, as well as upload your own files. Here are the instructions:

Install and login:
```bash
# install huggingface-hub
pip install huggingface-hub
# login
hf auth login
```

Download (Dataset):
```bash
# set the dataset id
export DATASET_ID=orca-dev-test
# download the dataset
# the dataset will be downloaded to ~/datasets/$DATASET_ID
hf download nvidia/$DATASET_ID --repo-type=dataset --local-dir ~/datasets/$DATASET_ID
```

Update & Upload:
```bash
# NOTE: please download first and ensure it is the latest version. See above for details
cd ~/datasets/$DATASET_ID
# copy the file inside the directory to the dataset
# for example: 
# cp <file_name> ./<path>/
hf upload nvidia/$DATASET_ID . --repo-type=dataset
```

Available dataset ids:
- `orca-dev-test`
  - prototyping datasets for ORCA pipeline dev and test
