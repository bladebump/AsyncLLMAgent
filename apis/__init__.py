from .law import law_router
from .files import files_router
from .event import event_router

all_routers = [law_router, files_router, event_router]

__all__ = ["all_routers"]
