# core/clima.py
from datetime import datetime

def is_frio() -> bool:
    """Simula clima frio às sextas à noite."""
    agora = datetime.now()
    return agora.weekday() == 4 and 18 <= agora.hour <= 23