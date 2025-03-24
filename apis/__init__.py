from .law import law_router
from .files import files_router

all_routers = [law_router, files_router]

__all__ = ["all_routers"]
