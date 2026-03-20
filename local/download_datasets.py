# this does not work yet

from datasets import load_dataset, Audio
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
DATASET_URL = "openslr/librispeech_asr"


dataset = load_dataset(DATASET_URL)

dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

for split in ["train", "validation", "test"]:
    dataset[split].save_to_disk(DATA_DIR / DATASET_URL / split)
