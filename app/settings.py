import json
from pathlib import Path
import logging

SETTINGS_FILE = Path("settings.json")


def load_settings():
    """
    Loads settings from the JSON file.
    If the file doesn't exist, returns default settings.
    """
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logging.error(f"Error loading settings file: {e}. Using defaults.")
            return get_default_settings()
    else:
        return get_default_settings()


def save_settings(settings):
    """
    Saves the given settings dictionary to the JSON file.
    """
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=4)
    except IOError as e:
        logging.error(f"Error saving settings file: {e}")


def get_default_settings():
    """
    Returns a dictionary with the default settings.
    """
    return {
        "camera_id": 0,
        "background_path": "assets/default_backgrounds/office.jpg",
        "blur": False,
    }
