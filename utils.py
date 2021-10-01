
import yaml

def load_settings(filename="settings.yml"):
    with open(filename, "r") as f:
        settings = yaml.safe_load(f)
    return settings

