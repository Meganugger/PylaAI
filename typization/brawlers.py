try:
    from enum import StrEnum
except ImportError:
    from enum import Enum

    class StrEnum(str, Enum):
        pass


class BrawlerName(StrEnum):
    Shelly = 'shelly'
    Larry = 'larrylawrie'
