import sys
from pathlib import Path


def setup_examples_test(add_path: str = ""):
    wkdir = Path(__file__).parent.parent.parent.parent / "examples" / add_path
    print(wkdir)
    sys.path.insert(0, str(wkdir))
    return wkdir


def setup_docs_test(add_path: str = ""):
    wkdir = Path(__file__).parent.parent.parent.parent / "docs_src" / "pages" / add_path
    print(wkdir)
    sys.path.insert(0, str(wkdir))
    return wkdir
