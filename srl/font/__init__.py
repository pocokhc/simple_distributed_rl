import os

# windows     : C:\Windows\Fonts
# max         : /System/Library/Fonts
# linux       : /usr/local/share/fonts/
# colaboratory: /usr/share/fonts/


def get_font_path() -> str:
    font_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "FiraCode-Regular.ttf"))
    assert os.path.isfile(font_path), f"font file is not found({font_path})"
    return font_path
