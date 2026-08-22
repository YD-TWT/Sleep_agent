import json
from pathlib import Path

from GDesigner.utils.log import logger

class JSONLReader:

    @staticmethod
    def parse_file(file_path: Path) -> list:
        logger.info(f"Reading JSON Lines file from {file_path}.")
        with open(file_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]
