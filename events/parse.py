from pydantic import BaseModel
import base64

class Frame(BaseModel):
    timestamp: int
    data: str # base64编码的内容
    is_from_client: bool

class FrameParser:
    def __init__(self, frame_list: list[Frame]):
        self.frame_list = frame_list

    def parse(self):
        frame_list = []
        for frame in self.frame_list:
            if not frame.is_from_client:
                frame_list.append({
                    "timestamp": frame.timestamp,
                    "data": base64.b64decode(frame.data).decode("utf-8")
                })
        return frame_list
