from datetime import datetime
from constants import TIME_FORMAT


def get_time() -> str:
    return datetime.today().strftime(TIME_FORMAT)
