from .law import law_router
from .files import files_router
from .event import event_router
from .guide import guide_router
from .competition import competition_router
from .course import course_router

all_routers = [law_router, files_router, event_router, guide_router, competition_router, course_router]

__all__ = ["all_routers"]
