# Import V2 as default to maintain backwards compatibility
from .t5_datamodule_v2 import T5DataModule  # noqa: F401 - V2 with legacy default
from .t5_datamodule import T5DataModule as T5DataModuleV1  # noqa: F401 - Original
