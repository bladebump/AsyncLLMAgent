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
            frame_list.append({
                "timestamp": frame.timestamp,
                "data": base64.b64decode(frame.data).decode("utf-8"),
                "is_from_client": frame.is_from_client
            })
        return frame_list
    
class OutputParser:
    def parse(self):
        frame_list = []
        for frame in self.frame_list:
            if not frame.is_from_client:
                frame_list.append({
                    "timestamp": frame.timestamp,
                    "data": base64.b64decode(frame.data).decode("utf-8")
                })

class MergeParser(FrameParser):

    def __init__(self, frame_list: list[Frame]):
        super().__init__(frame_list)
        self.in_merge: bool = False
        self.merge_buffer: list[Frame] = []
        self.front_frame: Frame | None = None
        self.merge_buffer_cursor: int = 0

    def handle_special_key(self, key_data: str):
        if b'\x7f'.decode() in key_data:  # 退格键
            if self.merge_buffer_cursor > 0:
                self.merge_buffer.pop(self.merge_buffer_cursor - 1)
                self.merge_buffer_cursor -= 1
            return True
        
        # 处理方向键左右
        if b'\x1b[C'.decode() in key_data:  # 右方向键
            if self.merge_buffer_cursor < len(self.merge_buffer):
                self.merge_buffer_cursor += 1
            return True
        elif b'\x1b[D'.decode() in key_data:  # 左方向键
            if self.merge_buffer_cursor > 0:
                self.merge_buffer_cursor -= 1
            return True
        return False

    def decode_frame(self):
        frame_list = []
        for frame in self.frame_list:
            frame_list.append(Frame(
                timestamp=frame.timestamp,
                data=base64.b64decode(frame.data).decode("utf-8"),
                is_from_client=frame.is_from_client
            ))
        self.frame_list = frame_list

    def parse(self):
        self.decode_frame()
        frame_list = []
        for frame in self.frame_list:
            if (self.front_frame is None):
                self.front_frame = frame
                if not frame.is_from_client:
                    frame_list.append(frame)
                continue
            if frame.is_from_client:
                self.front_frame = frame
                continue
            if not self.in_merge:
                if frame.data == self.front_frame.data:
                    self.in_merge = True
                    self.merge_buffer.insert(self.merge_buffer_cursor, frame)
                    self.merge_buffer_cursor += 1
                else:
                    frame_list.append(frame)
            elif self.in_merge:
                has_handle = self.handle_special_key(self.front_frame.data)
                if has_handle:
                    self.front_frame = frame
                    continue
                elif frame.data == self.front_frame.data:
                    self.merge_buffer.insert(self.merge_buffer_cursor, frame)
                    self.merge_buffer_cursor += 1
                else:
                    self.in_merge = False
                    if len(self.merge_buffer) > 0:
                        temp_frame = Frame(
                            timestamp=self.merge_buffer[0].timestamp,
                            data="".join([tmp_frame.data for tmp_frame in self.merge_buffer]),
                            is_from_client=False
                        )
                    frame_list.append(temp_frame)
                    frame_list.append(frame)
                    self.merge_buffer = []
                    self.merge_buffer_cursor = 0
            self.front_frame = frame
        if self.in_merge:
            temp_frame = Frame(
                timestamp=self.merge_buffer[0].timestamp,
                data="".join([tmp_frame.data for tmp_frame in self.merge_buffer]),
                is_from_client=False
            )
            frame_list.append(temp_frame)
        return [{"timestamp": frame.timestamp, "data": frame.data} for frame in frame_list]