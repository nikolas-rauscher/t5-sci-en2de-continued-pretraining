# test cleaning block

import json
from typing import Any, Dict, Iterable

from datatrove.data import Document # 

class StructMetaToJSON:
    """Wandelt Arrow-Struct im Feld 'metadata' in JSON-Text um."""
    def __call__(self, data: Iterable[Document], rank: int = 0, world_size: int = 1) -> Iterable[Document]:
        for doc in data:
            meta = doc.metadata
            if meta is not None and not isinstance(meta, str):
                doc.metadata = json.dumps(meta)
            yield doc