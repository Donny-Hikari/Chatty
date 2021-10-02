
import ruamel.yaml

yaml = ruamel.yaml.YAML()
yaml.preserve_quotes = True

def load_settings(filename="settings.yml"):
    with open(filename, "r") as f:
        settings = yaml.load(f)
    return settings

