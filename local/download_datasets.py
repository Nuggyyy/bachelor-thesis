from datasets import load_dataset, DatasetDict
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
DATASET_URL = "openslr/librispeech_asr"
DATASET_DIR = DATA_DIR / "librispeech_asr"
DATASET_DIR.mkdir(exist_ok=True)

dataset = DatasetDict()
dataset["train"] = load_dataset(DATASET_URL, "clean", split="train.360")
dataset["validation"] = load_dataset(DATASET_URL, "clean", split="validation")
dataset["test"] = load_dataset(DATASET_URL, "clean", split="test")

dataset["train"].save_to_disk(DATASET_DIR / "train")
dataset["validation"].save_to_disk(DATASET_DIR / "validation")
dataset["test"].save_to_disk(DATASET_DIR / "test")
