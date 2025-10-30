# NEuRT

TEXT

---

## Requirements

- Python 3.10 or higher
- Cuda 12.4 or higher
- Git (to clone the repository)

> Make sure Python 3.10+ is installed and added to your PATH.

---

## Installation

1. **Clone the repository:**

    ```bash
    > git clone https://github.com/Biomed-imaging-lab/NeuroBERT-dev.git
    > cd NeuroBERT-dev
    ```

2. **Run the setup script** to create a virtual environment and install dependencies:

   ### On Linux/macOS

    ```bash
    > bash setup.sh
    ```

   ### on Windows

    ```shell
    > setup.bat
    ```

    > This will create a `venv` directory and install all required Python packages from `requirements.txt`

## Running the scripts

There are two main scripts you can run:

### 1. Training/Testing classifier

```bash
> python -m scripts.train_cls scripts/example_config_cls.json [options]
```

**Optional arguments:**

- `-a / --array_format` : Data format, `"mem"` for memmap or `"arr"` for arrays (default: `"mem"`)
- `-m / --mode` : `"train"` or `"test"` (default: `"train"`)
- `-n / --num_workers` : Number of workers in dataloader (default: 2)
- `-s / --session_probs` : Count probabilities per session (0 or 1, default: 1)

### 2. Training/Testing reconstruction

```bash
> python -m scripts.train_recon scripts/example_config_recon.json [options]
```

**Optional arguments:**

- `-a / --array_format` : Data format, `"mem"` for memmap or `"arr"` for arrays (default: `"mem"`)
- `-m / --mode` : `"train"` or `"test"` (default: `"train"`)
- `-n / --num_workers` : Number of workers in dataloader (default: 2)
- `-p / --part_to_split` : Fraction of data used for training (e.g., `-p` 0.9 means 90% train / 10% validation, default: 0.9)

---

## License

This project is licensed under the [Apache License 2.0](LICENSE).
