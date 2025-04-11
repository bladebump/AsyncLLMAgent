from .parse import FrameParser, MergeParser, OutputParser, Frame
from enum import Enum
class ParserType(str, Enum):
    """解析器类型"""
    ALL = "all"
    MERGE = "merge"
    OUTPUT = "output"

class ParserFactory:

    @staticmethod
    def create_parser(parser_type: ParserType, frame_list: list[Frame], **kwargs) -> FrameParser:
        parsers = {
            ParserType.ALL: FrameParser,
            ParserType.MERGE: MergeParser,
            ParserType.OUTPUT: OutputParser
        }
        parser_class = parsers.get(parser_type)
        if not parser_class:
            raise ValueError(f"未知解析器类型: {parser_type}")
        return parser_class(frame_list, **kwargs)
