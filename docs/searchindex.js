Search.setIndex({"docnames": ["index", "pages/custom_algorithm", "pages/custom_env", "pages/install", "pages/quickstart"], "filenames": ["index.rst", "pages\\custom_algorithm.rst", "pages\\custom_env.rst", "pages\\install.rst", "pages\\quickstart.rst"], "titles": ["Welcome to SimpleDistributedRL's documentation!", "<span class=\"section-number\">4. </span>Create Original Algorithm", "<span class=\"section-number\">3. </span>Create Original Environment", "<span class=\"section-number\">1. </span>Installation", "<span class=\"section-number\">2. </span>Getting Started"], "terms": {"installation": 0, "getting": 0, "started": 0, "create": [0, 3, 4], "original": [0, 4], "environment": [0, 1], "\u7d22\u5f15": 0, "\u30e2\u30b8\u30e5\u30fc\u30eb": 0, "\u691c\u7d22": 0, "\u30da\u30fc\u30b8": 0, "\u3053\u3053": [1, 2, 3], "\u30d5\u30ec\u30fc\u30e0\u30ef\u30fc\u30af": [1, 2, 4], "\u81ea\u4f5c": 0, "\u4f5c\u6210": [1, 2, 4], "\u8aac\u660e": [0, 2], "\u307e\u3059": [1, 2, 3, 4], "\u69cb\u6210": [1, 2], "\u3068\u3057\u3066": [1, 2, 4], "\u5927\u304d\u304f": [1, 2], "\u4ee5\u4e0b": [1, 2, 3, 4], "\u3067\u3059": [1, 2, 3, 4], "openai": [2, 3], "base": [1, 2], "singleplay": 2, "turnbase": 2, "2player": 2, "\u767b\u9332": [0, 4], "(singleplay": 2, "\u30c6\u30b9\u30c8": [0, 1], "\u5b66\u7fd2": [0, 4], "\u306b\u3088\u308b": [0, 1], "\u73fe\u5728": [1, 2], "\u4e8c\u3064": 2, "\u3042\u308a": [1, 2, 4], "(openai": [], ")[": 1, "https": [2, 3, 4], ":/": [2, 3, 4], "github": [2, 3], ".com": [2, 3], "/openai": 2, "/gym": 2, "\u4ee5\u4e0bgym": 2, "\u7528\u610f": [1, 2], "\u3044\u308b": [1, 2, 4], "srl": [1, 2, 3, 4], ".base": [1, 2], ".env": [1, 2], ".envbase": 2, "\u7d99\u627f": [1, 2], "\u305d\u308c\u305e\u308c": [1, 2], "\u5834\u5408": [1, 2, 3, 4], "\u5225\u9014": [1, 2], "pip": [2, 3, 4], "install": [0, 2, 4], "\u5fc5\u8981": [1, 2, 3, 4], "\u307e\u305f": [1, 2, 3, 4], "\u4e00\u90e8": 2, "\u7279\u5316": 2, "\u5185\u5bb9": 2, "\u306a\u3089": [1, 2, 4], "\u306a\u3044": [1, 2, 4], "\u3042\u308b": [1, 2, 3, 4], "\u898b\u6bd4\u3079": 2, "\u3082\u3089\u3048\u308c": 2, "\u5206\u304b\u308a": 2, "\u30a2\u30eb\u30b4\u30ea\u30ba\u30e0": [0, 2, 3], "\u5411\u3051": 2, "\u60c5\u5831": [1, 2, 3], "\u5897\u3048": 2, "\u203bgym": 2, "\u66f4\u65b0": [1, 2], "\u306b\u3088\u308a": 2, "\u4e0d\u6b63": 2, "\u306a\u308b": [1, 2, 3], "\u53ef\u80fd": [1, 2], "\u3088\u308a": [1, 2], "\u6b63\u78ba": 2, "\u516c\u5f0f": 2, "(<": [], "\u898b\u3066": 2, "\u3060\u3055\u3044": [1, 2, 3], "\u5177\u4f53": [1, 2], "\u30b3\u30fc\u30c9": [1, 2], "import": [1, 2, 3, 4], "from": [1, 2, 3, 4], "spaces": 2, "numpy": [1, 2, 3, 4], "as": [1, 2, 4], "np": [1, 2, 4], "class": [1, 2], "mygymenv": 2, "(gym": 2, "):": [1, 2, 4], "\u3067\u304d\u308b": [1, 2, 4], "render": [1, 2], "_modes": 2, "\u6307\u5b9a": [1, 2, 4], "\u3088\u3046": [1, 2, 4], "metadata": 2, "{\"": 2, "\":": [1, 2, 3, 4], "[\"": [1, 2], "ansi": 2, "\",": [1, 2, 3, 4], "rgb": [2, 3, 4], "_array": [1, 2], "\"]": [1, 2], ", \"": 2, "_fps": 2, "def": [1, 2, 3, 4], "init": [1, 2], "__": [1, 2, 3, 4], "self": [1, 2], "_mode": 2, "str": [1, 2], "none": [1, 2], ".render": [1, 2, 4], "\"\"": [1, 2, 4], "2\u3064": 2, "\u5909\u6570": [1, 2], "\u5b9a\u7fa9": [1, 2], ".space": 2, "\u7701\u7565": 2, ".action": [1, 2], "_space": [2, 4], "\u30a2\u30af\u30b7\u30e7\u30f3": [1, 2], "\u53d6\u308a\u3046\u308b": 2, "\u7bc4\u56f2": 2, ".observation": 2, "\u72b6\u614b": [1, 2], ".discrete": [1, 2], ".box": 2, "(-": 2, "shape": 2, "=(": 2, ",)": 2, "reset": 2, "(self": [1, 2], "*,": 2, "seed": 2, "=none": [1, 2, 4], "options": 2, ")-": 2, "tuple": [1, 2], "[np": [1, 2], ".ndarray": [1, 2], "dict": [1, 2, 4], "]:": [1, 2], "super": [1, 2], "()": [1, 2, 3, 4], "(seed": 2, "=seed": 2, "\u30a8\u30d4\u30bd\u30fc\u30c9": [1, 2, 4], "\u6700\u521d": [1, 2], "\u5b9f\u884c": [1, 2, 3, 4], "\u3002\uff08": [1, 2], "\u521d\u671f\u5316": [1, 2], "\u51e6\u7406": [1, 2], "return": [1, 2, 4], "\u521d\u671f": [1, 2], ".array": [1, 2], "([": [1, 2], "],": [1, 2], "dtype": 2, "=np": 2, ".float": 2, "{}": 2, "step": [1, 2], "action": [1, 2, 4], "->": [1, 2], "float": [1, 2], "bool": [1, 2, 4], "1step": [1, 2, 4], "\u9032\u3081\u308b": 2, "\u5373\u6642": 2, "\u5831\u916c": [1, 2, 4], "\u4e88\u5b9a\u901a\u308a": 2, "\u7d42\u4e86": [1, 2], "\u305f\u3089": 2, "true": 2, "(terminated": 2, "),": [1, 2, 4], "\u4e88\u60f3": 2, "(truncated": 2, "\u4efb\u610f": [1, 2], "false": 2, "\u63cf\u753b": [1, 2, 3, 4], "\u66f8\u304d": 2, "pass": [1, 2], "\u767b\u9332\u6642": 2, "id": 2, "\u540d\u524d": [1, 2], "-vx": 2, "\u3068\u3044\u3046": 2, "\u5f62\u5f0f": [1, 2, 4], ".envs": 2, ".registration": [1, 2], ".register": 2, "=\"": [2, 4], "-v": [2, 4], "entry": [1, 2], "_point": [1, 2], "=_": 2, "name": [1, 2, 3, 4], "max": [1, 2], "_episode": [1, 2], "_steps": [1, 2], "\u547c\u3073\u51fa\u305b": 2, "env": [1, 2, 4], ".make": 2, "(\"": [1, 2, 3, 4], "\")": [1, 2, 4], "observation": 2, ".reset": [1, 2], "for": [1, 2, 4], "in": [1, 2], "range": [1, 2], "reward": [1, 2, 4], "terminated": 2, "truncated": 2, "info": [1, 2], ".step": [1, 2], "(env": [1, 2, 4], ".sample": [1, 2], "if": [1, 2, 3, 4], "or": [2, 3, 4], ".close": 2, "\u5171\u901a": 2, "\u4f7f\u7528": [1, 2, 3], "\u57fa\u5e95": [1, 2], "\u4f7f\u3046": [1, 2, 3], "\u3053\u306e": [1, 2], "\u305f\u3060": 2, "\u8907\u6570": [1, 2], "\u306b\u3044\u304f\u3064\u304b": 2, "\u4f59\u5206": 2, "\u5165\u3063": 2, "\u3053\u308c": 2, "\u3069\u3046": [1, 2], "\u3044\u3063": 2, "\u3082\u306e": 2, "\u7a0b\u5ea6": 2, "\u6c7a\u5b9a": [1, 2], "\u305d\u308c": [1, 2], "\u304b\u3089": [1, 2, 3], "\u9078\u3093": 2, "\u304f\u3060": [1, 2, 4], "\u3055\u3044": [1, 2, 4], "\u4f5c\u308b": 2, "singleplayenv": 2, "\u4e8c\u4eba": 2, "\u95a2\u6570": 2, "\u30fb\u30d7\u30ed\u30d1\u30c6\u30a3": [1, 2], "typing": [1, 2], "any": [1, 2], ".spaces": 2, "spacebase": 2, ".define": [1, 2], "envactiontype": 2, "envobservationtype": 2, "envobservationtypes": 2, "infotype": 2, "myenvbase": 2, "(envbase": 2, "property": [1, 2], "\u8fd4\u3057": [1, 2, 4], "(spacebase": 2, "\u5f8c\u8ff0": [1, 2], "raise": [1, 2], "notimplementederror": [1, 2], "_type": [1, 2], "\u7a2e\u985e": [1, 2], "\u5217\u6319": 2, "discrete": [1, 2], "\u96e2\u6563": [1, 2], "continuous": [1, 2], "\u9023\u7d9a": [1, 2], "gray": 2, "2ch": 2, "\u30b0\u30ec\u30fc": 2, "\u753b\u50cf": 2, "3ch": 2, "color": 2, "\u30ab\u30e9\u30fc": 2, "\u6b21\u5143": 2, "\u7a7a\u9593": 2, "int": [1, 2], "\u6700\u5927": 2, "\u30b9\u30c6\u30c3\u30d7": 2, "player": [1, 2], "_num": [1, 2], "\u30d7\u30ec\u30a4\u30e4\u30fc": [1, 2], "\u4eba\u6570": 2, "[envobservationtype": 2, "1\u30a8\u30d4\u30bd\u30fc\u30c9": 2, "returns": [1, 2], "_state": [1, 2], "list": [1, 2], "[float": [1, 2], "args": [1, 2], "envaction": 2, "_index": [1, 2], "next": [1, 2], "...": [1, 2, 4], "done": [1, 2, 4], "episode": 2, "\u305f\u304b": 2, "_player": 2, "\u30d7\u30ec\u30a4\u30e4\u30fcindex": 2, "backup": 2, "/restore": 2, "\u73fe\u74b0": 2, "\u5fa9\u5143": [1, 2], "restore": 2, "data": [1, 2], ".genre": 2, ".singleplay": 2, ".singleplayenv": 2, "mysingleplayenv": 2, "(singleplayenv": 2, "call": [1, 2], "_reset": [1, 2], "envobservation": 2, "_step": [1, 2], "\u304b\u3069\u3046": [1, 2], ".turnbase": 2, "myturnbase": 2, "(turnbase": 2, "\u756a\u53f7": [1, 2], "\u8fd4\u3059": [1, 2], "\u6c7a\u3081\u308b": [1, 2], "\u306a\u308a": [1, 2, 3], "\u73fe\u72b6": [1, 2], "discretespace": 2, "(int": [], "1\u3064": 2, "\u6574\u6570": 2, "\u8868\u3057": 2, "\u4f8b\u3048": 2, "\u3068\u3057": 2, "\u53d6\u308a": [1, 2], "arraydiscretespace": 2, "(list": [], "[int": [1, 2], "])": [1, 2], "\u914d\u5217": [1, 2], "low": 2, "=-": 2, "high": 2, "\u3001[": 2, "continuousspace": 2, "(float": [], "\u5c0f\u6570": [1, 2], "(low": 2, "\u3001-": 2, "arraycontinuousspace": 2, "boxspace": 2, "(np": 1, "\u53d6\u308a\u6271\u3044": 2, "(shape": 2, "[-": [1, 2, 4], "]]": [1, 2], "\u8a73\u7d30": [1, 2], "\u30a4\u30f3\u30b9\u30c8\u30fc\u30eb": 3, "\u30c0\u30a6\u30f3\u30ed\u30fc\u30c9": 3, "\u3057\u3066": 3, "\u3053\u3068": [1, 3, 4], "\u3067\u304d": [1, 2, 3, 4], "\u76f4\u63a5": [1, 3], "\u3059\u308b": [0, 3, 4], "\u30b3\u30de\u30f3\u30c9": 3, "git": 3, "+https": 3, "/pocokhc": 3, "/simple": 3, "_distributed": 3, "_rl": 3, "\u65b9\u6cd5": [1, 3], "clone": 3, ".git": 3, "cd": 3, "simple": 3, "\u52d5\u4f5c": [1, 3, 4], "\u78ba\u8a8d": 3, "\u30b5\u30f3\u30d7\u30eb\u30b3\u30fc\u30c9": 3, "runner": [1, 2, 3, 4], "--": [1, 2, 4], "algorithm": [0, 3, 4], "load": [0, 3], "grid": [1, 3, 4], "isort": [], "skip": [], "noqa": 2, ".algorithms": [1, 2, 3, 4], "ql": [2, 3, 4], "main": [3, 4], "config": 4, ".config": [1, 2, 3, 4], "train": [1, 3, 4], "parameter": 4, "_,": [], ".train": [1, 2, 3, 4], "(config": 1, "timeout": [], "evaluate": 3, "rewards": [1, 2, 3, 4], ".evaluate": [1, 2, 3, 4], "_episodes": [1, 2, 4], "print": [1, 2, 3, 4], "(f": [2, 3, 4], "\"evaluate": [3, 4], "episodes": [3, 4], "}\"": [1, 2, 3, 4], "\"__": [3, 4], "\u30c7\u30a3\u30ec\u30af\u30c8\u30ea": 3, "\u30d1\u30b9": 3, "\u901a\u3063": 3, "\u3044\u308c": 3, "\u3060\u3051": [1, 3], "\u3067\u3082": [1, 3], "\u4f7f\u3048": 3, "os": 3, "sys": 3, "assert": [1, 3], ".path": 3, ".isdir": 3, "./": 3, "/srl": 3, "/\"": 3, "\u60f3\u5b9a": [1, 3], ".insert": 3, "(srl": [1, 2, 3], "._": [1, 3], "version": 3, "make": [2, 4], "sure": [], "your": [], "is": [], "off": [], "while": [1, 2], "installing": [], "the": 4, "camera": [], "module": [], "although": [], "it": [], "possible": [], "to": [], "on": 1, "this": [], "isn": [], "'t": [], "good": [], "practice": [], "active": [], "when": [], "removed": [], "'s": [], "damage": [], ").": [], "connect": [], "csi": [], "port": [], "raspberry": [], "long": [], "thin": [], "adjacent": [], "hdmi": [], "socket": [], "gently": [], "lift": [], "collar": [], "top": [], "of": 0, "comes": [], "don": [], "worry": [], "you": [], "can": [], "push": [], "back": [], "but": [], "try": [], "be": [], "more": [], "gentle": [], "future": [], "!)": [], "slide": [], "ribbon": [], "cable": [], "into": [], "with": [], "blue": [], "side": [], "facing": [], "ethernet": [], "where": [], "would": [], "'ve": [], "got": [], "model": [], "/a": [], "+)": [], "once": [], "seated": [], "press": [], "down": [], "lock": [], "place": [], "properly": [], "should": [], "able": [], "easily": [], "by": [], "without": [], "falling": [], "out": [], "following": [], "illustrations": [], "show": [], "well": [], "-seated": [], "correct": [], "orientation": [], "sat": [], "anything": [], "conductive": [], ".g": [], "usb": [], "ports": [], "its": [], "gpio": [], "pins": [], "includes": [], "small": [], "form": [], "-factor": [], "which": [], "requires": [], "adapter": [], "attach": [], "remove": [], "existing": [], "lifting": [], "and": 4, "pulling": [], "insert": [], "wider": [], "end": [], "conductors": [], "same": [], "direction": [], "lens": [], "finally": [], "at": [], "edge": [], "board": [], "careful": [], "they": [], "are": [], "delicate": [], "than": [], "collars": [], "regular": [], "inserting": [], "smaller": [], "setup": [], "look": [], "something": [], "like": [], "now": [], "apply": [], "power": [], "booted": [], "start": [], "configuration": [], "utility": [], "enable": [], "will": [], "need": [], "reboot": [], "after": [], "doing": [], "one": [], "-time": [], "so": [], "won": [], "do": [], "again": [], "unless": [], "re": [], "-install": [], "operating": [], "system": [], "switch": [], "sd": [], "cards": [], "rebooted": [], "terminal": 2, "command": [], "else": [1, 2], "happens": [], "read": [], "error": [], "message": [], "displayed": [], "recommendations": [], "suggested": [], "such": [], "messages": [], "reboots": [], "soon": [], "run": 0, "supply": [], "insufficient": [], "running": [], "plus": [], "whatever": [], "other": 0, "peripherals": [], "have": [], "attached": [], "\u4f7f\u3044": [1, 2], "\u5f15\u6570": [1, 2, 4], "\u5099\u8003": 2, "-|": [], "|id": [], "\u30e6\u30cb\u30fc\u30af": [1, 2], "\u88ab\u3089": 2, "\u306a\u3051\u308c": 2, "\u7279\u306b": [1, 2], "\u5236\u9650": 2, "\u307e\u305b": [1, 2], "|entry": [], "|`": [], "\u30e2\u30b8\u30e5\u30fc\u30eb\u30d1\u30b9": [1, 2], ":\"": [1, 2], "`|": [], "importlib": [1, 2], ".import": [1, 2], "_module": [1, 2], "\u547c\u3073\u51fa\u305b\u308b": [1, 2], "|kwargs": [], "\u751f\u6210": 2, "||": [], "kwargs": [1, 2, 4], "aa": [], "\u4ed6\u5fc5": 2, "\u8a2d\u5b9a": [1, 2, 4], "\u8ffd\u52a0": [1, 2], "_info": 2, "\u306b\u95a2\u3059\u308b": [1, 2], "min": 2, "baseline": 2, "close": 2, "get": [1, 2], "_invalid": [1, 2], "_actions": [1, 2], "\u7121\u52b9": 2, "_get": 2, "set": [1, 2], "_seed": 2, "optional": [1, 2], "\u4e71\u6570": 2, "ai": 2, "_worker": 2, ".rl": [1, 2], ".workerbase": 2, "_terminal": [1, 2, 4], "**": [1, 2], "\u72b6\u6cc1": 2, "\u8868\u793a": [1, 2, 3, 4], "_rgb": [1, 2], "_to": [1, 2], "_str": [1, 2], "union": 2, "[str": 2, "\u6587\u5b57\u5217": 2, "\u5909\u63db": 2, "(action": [1, 2], "_interval": 2, "\u901f\u5ea6": 2, "_key": 2, "_bind": [2, 4], "keybindtype": 2, "\u30ad\u30fc": 2, "\u914d\u7f6e": 2, "\u7d10\u3065\u3051\u308b": 2, "registration": 2, "sampleenv": 2, "={": 2, "},": 2, "\u5de6\u53f3": 2, "\u52d5\u3051": 2, "\u30b4\u30fc\u30eb": 2, "\u30b7\u30f3\u30d7\u30eb": [2, 4], "\u3068\u308a": 2, "\u3042\u3048": 2, "\u52d5\u304b\u3059": 2, "sample": [1, 2, 4], "_env": 2, "\u30d5\u30a1\u30a4\u30eb": 2, "state": [1, 2], "(render": 2, "total": 2, "_reward": 2, "not": [1, 2], ".done": [1, 2], ".reward": [1, 2], "\"step": 2, "(total": 2, "\u6700\u4f4e\u9650": 2, "\u3061\u3083\u3093": 2, "\u52d5\u304f": 2, ".test": [1, 2], "testenv": 2, "tester": [1, 2], ".play": [2, 4], "_test": 2, "_config": [1, 2, 4], "=srl": [], ".envconfig": [1, 2, 4], "rl": 4, "=ql": 2, "# q": 2, "memory": 1, "history": [], "\u5e73\u5747": [1, 2], "\u7d50\u679c": [1, 2, 4], ".mean": [1, 2, 4], "(rewards": [1, 2, 4], "))": [1, 2], "(reward": [], ".animation": [1, 2, 4], ".create": [], "_anime": [], "(scale": [], "save": 4, ".gif": [1, 2, 4], ".display": [], "notebook": 4, "enum": 2, "dataclasses": [1, 2], "dataclass": [1, 2], "move": 2, ",\n)": 2, "(enum": 2, ".enum": 2, "left": 2, "right": 2, "post": 2, "_init": 2, ".field": 2, "(len": 2, ".player": [1, 2], "_pos": 2, "_:": 2, "_)": 2, ".left": 2, "elif": 2, ".right": 2, "[self": 2, "==": 2, ".move": 2, "= \"": 2, "[x": 2, ".\"": 2, "(s": [1, 2], "ndarray": 2, "spaceclass": 2, "\u6982\u8981": [0, 2], "\u6700\u3082": 4, "\u8a55\u4fa1": [2, 4], "\u5225\u3005": 4, "use": 4, "gym": [3, 4], "pygame": [3, 4], "path": 4, "_path": 4, "\"_": 4, "params": 4, ".dat": 4, "parameters": 4, "refer": 4, "argument": 4, "completion": 4, "code": 4, "frozenlake": 4, "_parameter": 4, "loads": [], "file": [], "exists": [], ".isfile": [], "(_": 4, ".load": 4, "sequence": 4, "training": [1, 4], "remote": [], "_memory": 1, "=parameter": [], "distributed": [1, 4], "_mp": 4, ".save": 4, "\"average": 4, ")}": [1, 4], "watch": [], "progress": [], "window": [], "opencv": [3, 4], "-python": [3, 4], "pillow": [3, 4], "matplotlib": [3, 4], "animation": 2, "_window": 4, "\u74b0\u5883": [0, 1, 3], "\u624b\u52d5": 4, "\u904a\u3076\u4f8b": [], "key": 4, "\u5165\u529b": 4, "\u30ab\u30b9\u30bf\u30de\u30a4\u30ba": [], "[atari": [], "accept": [], "-rom": [], "-license": [], "ale": 4, "/galaxian": 4, "=dict": 4, "(full": 4, "_action": [1, 2, 4], "=true": 4, "\"z": 4, ".k": 4, "_up": 4, "_right": 4, "_left": 4, "_down": 4, "_z": 4, "=key": 4, "q\u5b66\u7fd2": 1, "5\u3064": 1, "\u9023\u643a": 1, "\u73fe\u308c": 1, "\u3044\u307e\u305b": 1, "\u30cf\u30a4\u30d1\u30fc\u30d1\u30fc\u30d1\u30e9\u30e1\u30fc\u30bf": 1, "\u7ba1\u7406": 1, "\u304c\u3044": 1, "\u5f79\u5272": 1, "\u30cf\u30a4\u30d1\u30fc\u30d1\u30e9\u30e1\u30fc\u30bf": 1, "\u306a\u3069": 1, "\u30d1\u30e9\u30e1\u30fc\u30bf": 1, "\u30b5\u30f3\u30d7\u30eb": 1, "\u53ce\u96c6": 1, "\u9001\u4fe1": 1, "\u884c\u52d5": 1, "\u8aad\u3080": 1, "\uff08read": 1, "only": 1, "\u53d6\u5f97": 1, "\u4fdd\u6301": 1, "\u5206\u6563": 1, "\u975e\u540c": 1, "\u8996\u70b9": 1, "\u5927\u304d": 1, "\u9055\u3044": 1, "\u9001\u308b": 1, "\u30bf\u30a4\u30df\u30f3\u30b0": 1, "\u53d6\u308a\u51fa\u3059": 1, "\u540c\u671f": 1, "\u4ed5\u65b9": 1, "\u3044\u304d": 1, "\u5f37\u5316": 1, ".rlconfig": 1, "\u3067\u3053\u308c": 1, "\u30a4\u30f3\u30bf\u30d5\u30a7\u30fc\u30b9": 1, "\u304a\u308a": 1, "\u5f53\u3066": [], "\u307e\u308b": [], "\u305d\u3061\u3089": 1, "\u305f\u307b\u3046": [], "discreteactionconfig": [], "\u30e2\u30c7\u30eb\u30d5\u30ea\u30fc": 1, "dqn": 1, "continuousactionconfig": [], "ddpg": 1, "sac": 1, "rlconfig": 1, "rltypes": 1, ".processor": 1, "processor": 1, "\u5fc5\u305a": 1, "\u66f8\u3044": 1, "myconfig": 1, "(rlconfig": 1, "getname": 1, "\u30bf\u30a4\u30d7": 1, "\u96e2\u6563\u5024": 1, "\u9023\u7d9a\u5024": 1, "\u3069\u3061\u3089": 1, "\u53d7\u3051": 1, "\u53d6\u308b": 1, "option": 1, "\u5fc5\u9808": [1, 3], "_params": 1, "\u8a18\u8f09": 1, "\u547c\u3073\u51fa\u3057": 1, "_by": 1, "_actor": 1, "actor": 1, "_id": 1, "\u3068\u304d": 1, "\u547c\u3073\u51fa\u3055": 1, "\u95a2\u4fc2": [1, 3], "_processor": 1, "[processor": 1, "\u524d\u51e6": 1, "\u305f\u3044": 1, "\u4e0d\u8981": [], "\u3002(": 1, "\u56fa\u5b9a": [], "batch": 1, "\u6e21\u3059": 1, "\u6301\u3063": 1, "multiprocessing": 1, "\u30b5\u30fc\u30d0\u30d7\u30ed\u30bb\u30b9": 1, "manager": 1, "\u306e\u3067": [1, 3], "\u30a2\u30af\u30bb\u30b9": 1, "\u306a\u304f": [1, 4], "\u70b9\u3060\u3051": 1, "\u5236\u7d04": 1, "\u7d4c\u7531": 1, "\u3084\u308a": 1, "\u3088\u304f": 1, ".remote": 1, "\u9806\u5e8f": 1, "\u901a\u308a": 1, "\u53d6\u308a\u51fa\u3057": 1, "queue\u307f": 1, "\u30e9\u30f3\u30c0\u30e0": 1, "\u512a\u5148": 1, "\u9806\u4f4d": 1, "\u5f93\u3044": 1, "\u307e\u305a": 1, "rlremotememory": 1, "cast": [], "myremotememory": 1, "(rlremotememory": 1, "\u30b3\u30f3\u30c8\u30e9\u30af\u30bf": 1, "\u6e21\u3057": 1, ".__": 1, "(*": 1, "\u5165\u308a": 1, "(myconfig": 1, "length": 1, "\u30e1\u30e2\u30ea": 1, "\u4fdd\u5b58": [1, 4], "_restore": 1, "/call": 1, "_backup": 1, ".buffer": 1, "\u305d\u306e": [1, 3, 4], "\u597d\u304d": 1, "\u9806\u756a\u901a\u308a": 1, "add": 1, "\u308c\u308b": 1, "(sequenceremotememory": 1, "(none": 1, ".add": 1, "dat": 1, "(dat": 1, "\u30b5\u30a4\u30ba": [], "capacity": [], "(experiencereplaybuffer": 1, "\u5bb9\u91cf": [], "\u30bb\u30c3\u30c8": [], ".capacity": [], "\u5b9f\u884c\u4f8b": [], "\u5f93\u3063": 1, "update": 1, "\u5c11\u3057": [], "\u8907\u96d1": [], "field": [], ".memory": [], ".priority": 1, "_experience": 1, "_replay": 1, ".memories": 1, "replaymemoryconfig": 1, "(iprioritymemoryconfig": [], "(default": [], "_factory": [], "=lambda": [], "(priorityexperiencereplay": 1, "indices": 1, "batchs": 1, "weights": 1, "(batch": 1, "_size": 1, "(batchs": 1, ".update": 1, "(indices": 1, "11": 1, "\u3044\u304f\u3064\u304b": 1, "\u5207\u308a": 1, "\u66ff\u3048": 1, "\u540c\u3058": 1, "proportionalmemoryconfig": 1, "\u91cd\u8981": 1, "\u306b\u3088\u3063": 1, "\u78ba\u7387": 1, "\u5909\u308f\u308a": 1, "\u9ad8\u3044": 1, "\u307b\u3069": 1, "\u9078\u3070": 1, "\u4e0a\u304c\u308a": 1, "rankbasememoryconfig": 1, "\u30e9\u30f3\u30ad\u30f3\u30b0": 1, "\u4e0a\u304c\u308b": 1, "proportional": 1, "\u6df1\u5c64": 1, "\u30cb\u30e5\u30fc\u30e9\u30eb\u30cd\u30c3\u30c8\u30ef\u30fc\u30af": 1, "rlparameter": 1, "myparameter": 1, "(rlparameter": 1, "\u4ed6\u4efb": 1, "/worker": 1, "\u90e8\u5206": 1, "\u7d4c\u9a13": 1, "\u53d7\u3051\u53d6\u3063": 1, "rltrainer": 1, "mytrainer": 1, "(rltrainer": 1, ".parameter": 1, "(myparameter": [], "(myremotememory": [], "_train": 1, "_count": 1, "\"\"\"": 1, "\u56de\u6570": 1, "\u623b\u308a\u5024": 1, "\u8f9e\u66f8": 1, "\u5b9f\u969b": 1, "\u53c2\u7167": 1, "rlworker": [], "\u30d5\u30ed\u30fc": 1, "\u3059\u3054\u304f": 1, "\u7c21\u5358": 1, "\u66f8\u304f": 1, ".on": 1, ".policy": 1, "\u5408\u308f\u305b": 1, "\u57fa\u672c": 1, "\u4e0a\u8a18": 1, "\u4ee5\u5916": 1, "myworker": 1, "(discreteactionworker": 1, "_on": 1, "invalid": 1, "\u547c\u3070": 1, "\u521d\u671f\u72b6\u614b": 1, "\u6709\u52b9": 1, "\u30ea\u30b9\u30c8": 1, "_policy": 1, "\u30bf\u30fc\u30f3": 1, ".continuous": 1, "(continuousactionworker": 1, "[list": 1, "envrun": 1, "workerrun": 1, "\u64cd\u4f5c": 1, "\u51fa\u6765\u308b": 1, "\u591a\u3044": 1, "\u4ed5\u69d8": 1, "\u5909\u308f\u308b": 1, "\u6027\u5927": 1, "\u304d\u3044": 1, "\u4e00\u65e6": 1, "\u4fdd\u7559": 1, "todo": 1, ".modelbase": 1, ".worker": 1, "(modelbaseworker": 1, "[rlactiontype": 1, "rlaction": 1, "\u304b\u3069": 1, "\u63cf\u753b\u7528": 1, "\u81ea\u5206": 1, "[rlaction": 1, "\u9650\u5b9a": 1, "_valid": [], "valid": [], "\u4ee5\u964d": 1, "register": [1, 2], "json": 1, "random": 1, "(discreteactionconfig": [], "epsilon": [1, 4], "test": 1, "_epsilon": 1, "gamma": 1, "lr": 1, "myrl": 1, ".q": 1, "q\u5b66\u7fd2\u7528": 1, "\u30c6\u30fc\u30d6\u30eb": 1, ".loads": 1, "(data": 1, ".dumps": 1, "q\u5024": 1, "_values": 1, "[state": 1, "(parameter": [], "(remotememory": [], "td": 1, "_error": 1, "q\u30c6\u30fc\u30d6\u30eb": 1, "_s": 1, ".get": 1, "_q": 1, "(n": 1, "target": 1, ".gamma": 1, "[action": 1, "+=": 1, ".lr": 1, "len": 1, "\"q": 1, ".state": 1, "(state": [1, 4], ".tolist": 1, "\u63a2\u7d22": 1, "\u5909\u3048\u308b": 1, ".training": 1, ".epsilon": 1, ".random": 1, "\u4f4e\u3044": 1, "\u79fb\u52d5": 1, ".asarray": 1, "\u6700\u5927\u5024": 1, "\u9078\u3076": 1, "\u3042\u308c": 1, ".choice": 1, ".where": 1, "= q": 1, ".max": 1, "(next": 1, "\u53ef\u8996": 1, "\u5316\u7528": 1, "\u4eca\u56de": 1, "maxa": 1, ".argmax": 1, "*\"": 1, "= f": 1, "\"{": 1, "(a": 1, ": {": 1, "[a": 1, "5f": 1, "testrl": 1, ".simple": 1, "_check": 1, "=config": [], "(lr": 1, "-grid": 1, "##": [1, 4], "0s": 1, "work": [1, 4], "..": [1, 4], ". x": [1, 4], ".p": [1, 4], "\u2191:": 1, "(\u2191": [1, 4], "\u2190:": 1, "\u2192:": 1, "(\u2192": [1, 4], "pg": [1, 4], "\u30e9\u30a4\u30d6\u30e9\u30ea": 3, "\u5165\u308c": 3, "files": 3, "\u6a5f\u80fd": [3, 4], "\u306b\u3088\u3063\u3066": [3, 4], "tensorflow": [1, 3], "-addons": 3, "-probability": 3, "torch": [1, 3], "pytorch": 3, ".org": [3, 4], "/get": 3, "-started": 3, "/locally": 3, "\u7d71\u8a08": 3, "\u6271\u3046": 3, "pandas": 3, "profile": 3, "psutil": 3, "pynvml": 3, "\u9664\u3044": 3, "\u4e00\u62ec": 3, "download": 0, "(no": 0, "using": 0, "library": 0, "basic": 0, "study": 0, "commonly": 0, "example": 0, "manual": [], "play": [], "\u5b9f\u88c5": 0, "\u30af\u30e9\u30b9": 0, "space": 0, "\u306b\u3064\u3044\u3066": 0, "\u5b9f\u88c5\u4f8b": 0, "(q": 0, "_run": 1, "how": 0, "/gymnasium": 4, "\u8aad\u307f\u8fbc\u307f": [], "'gymnasium": [], "\u8aad\u307f\u8fbc\u3080": [], "gymnasium": [2, 3, 4], "replay": [], "\u5bfe\u5fdc": 4, "\u3053\u3053\u3067": [], "\u30ec\u30dd\u30b8\u30c8\u30ea": [], "(gymnasium": [], "/farama": [], "-foundation": [], "\u307e\u307e": 4, "\u4f7f\u308f": [], "-retro": 4, "support": 4, "python": 4, "<=": 4, "retro": 4, "airstriker": 4, "-genesis": 4, "level": 4, "_make": 4, "_func": 4, "=retro": 4, "external": 0, "methods": 0, "(load": 4, "= _": [], "=false": 4, "axis": 4, ".replay": 4, "\u51fa\u6765": 4, "\u8aad\u307f\u8fbc\u3093": 4, "\u30b7\u30df\u30e5\u30ec\u30fc\u30b7\u30e7\u30f3": 4, "\u5f8c\u304b\u3089": [], "\u898b\u8fd4\u3059": 4, "\u69d8\u5b50": 4, "\u51fa\u529b": 4, "\u30a8\u30d4\u30bd\u30fc\u30c9\u30b7\u30df\u30e5\u30ec\u30fc\u30b7\u30e7\u30f3": 4, "\u6620\u50cf": 4, "\u6b8b\u305b\u308b": 4, "\u30c7\u30fc\u30bf": 4, "\u306a\u304c\u3089": 4, "\u30a2\u30cb\u30e1\u30fc\u30b7\u30e7\u30f3": 4, "\u3001'": 4, ".artistanimation": 4, "\u30d7\u30ec\u30a4": 4, "'key": 4, "\u904a\u3079": 4, "\u3068\u3088\u308a": 4, "\u3042\u3063": 4, "_observation": 1, ":myconfig": 1, "envworker": 1, "\u30eb\u30fc\u30eb\u30d9\u30fc\u30b9": 1, "extendworker": 1, "\u6df7\u305c": 1, "modelbaseworker": 1, "rendering": 1, "\u6700\u5927steps": 1, ", config": 1, ".value": 2, "\u2190\"": 2, "\u2192\"": 2, ", rl": 2, "\u53ef\u8996\u5316": 2, "\u30b3\u30f3\u30b9\u30c8\u30e9\u30af\u30bf": 1, "_use": 1, "_framework": 1, "\u3082\u306a\u3057": 1, ".sequence": 1, ".experience": 1, "_buffer": 1, "experiencereplaybufferconfig": 1, "\u52a0\u3048": 1, "\u9806\u756a": 1, "priorityexperiencereplayconfig": 1, ".runner": [1, 2, 3, 4], "(timeout": [1, 2, 3, 4], "(max": [1, 2, 4], "_save": [1, 2, 4], "_gif": [1, 2, 4], "_scale": [1, 2], "_runner": 4, "\u898b\u308b": 4, "(\u2190": 4, "':": 4, "_display": 4, "atari": 4, "also": 4, "see": 4, "url": 4, "below": 4, ".farama": 4, "/environments": 4, "/atari": 4, "(key": 4}, "objects": {}, "objtypes": {}, "objnames": {}, "titleterms": {"welcome": 0, "to": [0, 4], "simpledistributedrl": 0, "'s": 0, "documentation": 0, "contents": 0, "indices": 0, "and": 0, "tables": 0, "create": [1, 2], "original": [1, 2], "environment": [2, 4], "\u5b9f\u88c5": [1, 2], "\u30af\u30e9\u30b9": [1, 2], "gym": 2, "\u74b0\u5883": [2, 4], "\u5229\u7528": 2, "envbase": 2, "\u3059\u308b": [1, 2], "\u65b9\u6cd5": [2, 4], "\u4e00\u4eba": 2, "\u30d7\u30ec\u30a4": 2, "\u30bf\u30fc\u30f3": 2, "space": 2, "\u306b\u3064\u3044\u3066": 2, "installation": 3, "download": 3, "(no": 3, "install": 3, "getting": 4, "started": 4, "pi": [], "zero": [], "testing": [], "\u81ea\u4f5c": [1, 2], "\u767b\u9332": [1, 2], "\u305d\u306e": 2, "\u30aa\u30d7\u30b7\u30e7\u30f3": 2, "\u5b9f\u88c5\u4f8b": [1, 2], "\u30c6\u30b9\u30c8": 2, "\u5b66\u7fd2": [1, 2], "\u306b\u3088\u308b": 2, "basic": 4, "run": 4, "of": 4, "study": 4, "commonly": 4, "example": 4, "manual": 4, "play": 4, "algorithm": 1, "\u6982\u8981": 1, "\u8aac\u660e": 1, "config": 1, "remotememory": 1, "sequenceremotememory": 1, "experiencereplaybuffer": 1, "priorityexperiencereplay": 1, "iprioritymemoryconfig": 1, "parameter": 1, "trainer": 1, "worker": 1, "discreteactionworker": 1, "continuousactionworker": 1, "modelbaseworker": [], "\u5171\u901a": 1, "\u30d7\u30ed\u30d1\u30c6\u30a3": 1, "\u95a2\u6570": [1, 4], "\u30a2\u30eb\u30b4\u30ea\u30ba\u30e0": 1, "(q": 1, "using": 3, "library": 3, "how": 4, "use": [], "/gymnasium": [], "\u8aad\u307f\u8fbc\u307f": [], "'gymnasium": 4, ".make": 4, "\u4ee5\u5916": 4, "\u8aad\u307f\u8fbc\u3080": 4, "load": 4, "external": 4, "other": 4, "methods": 4, "evaluate": 4, "replay": 4, "window": 4, "render": 4, "terminal": 4, "animation": 4, "rlworker": 1}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 8, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx": 57}, "alltitles": {"Welcome to SimpleDistributedRL's documentation!": [[0, "welcome-to-simpledistributedrl-s-documentation"]], "Contents": [[0, null]], "Indices and tables": [[0, "indices-and-tables"]], "Create Original Algorithm": [[1, "create-original-algorithm"]], "\u6982\u8981": [[1, "id1"]], "\u5b9f\u88c5\u3059\u308b\u5404\u30af\u30e9\u30b9\u306e\u8aac\u660e": [[1, "id2"]], "Config": [[1, "config"]], "RemoteMemory": [[1, "remotememory"]], "SequenceRemoteMemory": [[1, "sequenceremotememory"]], "ExperienceReplayBuffer": [[1, "experiencereplaybuffer"]], "PriorityExperienceReplay": [[1, "priorityexperiencereplay"]], "IPriorityMemoryConfig": [[1, "iprioritymemoryconfig"]], "Parameter": [[1, "parameter"]], "Trainer": [[1, "trainer"]], "Worker": [[1, "worker"]], "DiscreteActionWorker": [[1, "discreteactionworker"]], "ContinuousActionWorker": [[1, "continuousactionworker"]], "RLWorker": [[1, "rlworker"]], "Worker\u5171\u901a\u306e\u30d7\u30ed\u30d1\u30c6\u30a3\u30fb\u95a2\u6570": [[1, "id3"]], "\u81ea\u4f5c\u30a2\u30eb\u30b4\u30ea\u30ba\u30e0\u306e\u767b\u9332": [[1, "id4"]], "\u5b9f\u88c5\u4f8b(Q\u5b66\u7fd2)": [[1, "q"]], "Create Original Environment": [[2, "create-original-environment"]], "\u5b9f\u88c5\u30af\u30e9\u30b9": [[2, "id1"]], "gym \u306e\u74b0\u5883\u3092\u5229\u7528": [[2, "id2"]], "EnvBase\u30af\u30e9\u30b9\u3092\u5229\u7528\u3059\u308b\u65b9\u6cd5": [[2, "envbase"]], "\u4e00\u4eba\u30d7\u30ec\u30a4\u7528\u306e\u30af\u30e9\u30b9": [[2, "id3"]], "\u30bf\u30fc\u30f3\u5236\u4e8c\u4eba\u30d7\u30ec\u30a4\u7528\u306e\u30af\u30e9\u30b9": [[2, "id4"]], "\u305d\u306e\u4ed6\u30aa\u30d7\u30b7\u30e7\u30f3": [[2, "id5"]], "Space\u30af\u30e9\u30b9\u306b\u3064\u3044\u3066": [[2, "space"]], "\u81ea\u4f5c\u74b0\u5883\u306e\u767b\u9332": [[2, "id6"]], "\u5b9f\u88c5\u4f8b": [[2, "id7"]], "\u30c6\u30b9\u30c8": [[2, "id8"]], "Q\u5b66\u7fd2\u306b\u3088\u308b\u5b66\u7fd2": [[2, "q"]], "Installation": [[3, "installation"], [3, "id1"]], "Download(No install)": [[3, "download-no-install"]], "Using library": [[3, "using-library"]], "Getting Started": [[4, "getting-started"]], "Basic run of study": [[4, "basic-run-of-study"]], "Commonly run Example": [[4, "commonly-run-example"]], "How to load external environment": [[4, "how-to-load-external-environment"]], "'gymnasium.make' \u4ee5\u5916\u306e\u95a2\u6570\u3067\u74b0\u5883\u3092\u8aad\u307f\u8fbc\u3080\u65b9\u6cd5": [[4, "gymnasium-make"]], "Other Run Methods": [[4, "other-run-methods"]], "Evaluate": [[4, "evaluate"]], "Replay Window": [[4, "replay-window"]], "Render Terminal": [[4, "render-terminal"]], "Render Window": [[4, "render-window"]], "Animation": [[4, "animation"]], "Manual play Terminal": [[4, "manual-play-terminal"]], "Manual play Window": [[4, "manual-play-window"]]}, "indexentries": {}})