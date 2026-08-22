from dotenv import load_dotenv

from GDesigner.utils.const import GDesigner_ROOT

def load_env() -> None:
    load_dotenv(GDesigner_ROOT / "template.env")
