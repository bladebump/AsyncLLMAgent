from pydantic import BaseModel
import base64

class Frame(BaseModel):
    timestamp: int
    data: str # base64编码的内容
    is_from_client: bool

class Event(BaseModel):
    event_name: str
    event_input: str
    event_output: str
    event_info: str
    event_special: str
    event_start: int
    event_end: int

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
    
class OutputParser(FrameParser):
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
        self.command_context: str = ""  # 记录当前上下文

    def handle_special_key(self, key_frame: Frame):
        # 处理退格键
        if b'\x7f'.decode() in key_frame.data:
            if self.merge_buffer_cursor > 0:
                self.merge_buffer.pop(self.merge_buffer_cursor - 1)
                self.merge_buffer_cursor -= 1
            return True
        # 处理方向键左右
        elif b'\x1b[C'.decode() in key_frame.data:  # 右方向键
            if self.merge_buffer_cursor < len(self.merge_buffer):
                self.merge_buffer_cursor += 1
            return True
        elif b'\x1b[D'.decode() in key_frame.data:  # 左方向键
            if self.merge_buffer_cursor > 0:
                self.merge_buffer_cursor -= 1
            return True
        return False

    def handle_up_down(self, key_data: str, next_frame_data: str):
        up_key = b'\x1b[A'.decode()
        down_key = b'\x1b[B'.decode()
        
        # 检查是否是上下方向键
        is_up_down = up_key in key_data or down_key in key_data
        if not is_up_down:
            return False
            
        # 检查按上下键后的输出是否表明这是命令历史操作
        # 如果下一帧的输出是一个完整命令行且与当前上下文相关，则很可能是历史命令
        # 简单判断：下一帧输出内容应该看起来像一个命令而不是特殊界面的刷新
        
        # 1. 如果下一帧包含终端特殊界面刷新控制序列，则不是历史命令操作
        control_sequences = [
            b'\x1b[H'.decode(),  # 光标移到屏幕左上角（通常是全屏刷新的开始）
            b'\x1b[2J'.decode(),  # 清屏
            b'\x1b[?25l'.decode(),  # 隐藏光标
        ]
        for seq in control_sequences:
            if seq in next_frame_data:
                return False
                
        # 2. 如果下一帧的输出是一个看起来像命令的字符串，则可能是历史命令
        # 这是一个简化的判断，可能需要根据实际情况调整
        looks_like_command = (
            not next_frame_data.strip().startswith("\x1b") and  # 不以控制字符开始
            len(next_frame_data.split("\n")) <= 2 and  # 通常不会是多行输出
            len(next_frame_data) < 200  # 输出不会太长
        )
        
        return looks_like_command

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
            
            if self.handle_up_down(self.front_frame.data, frame.data):
                self.in_merge = True
                self.merge_buffer = [
                    Frame(
                        timestamp=frame.timestamp,
                        data=i,
                        is_from_client=frame.is_from_client
                    ) for i in frame.data
                ]
                self.merge_buffer_cursor = len(self.merge_buffer)
                self.front_frame = frame
                continue

            if not self.in_merge:
                if frame.data == self.front_frame.data:
                    self.in_merge = True
                    self.merge_buffer.insert(self.merge_buffer_cursor, frame)
                    self.merge_buffer_cursor += 1
                else:
                    if self.front_frame.is_from_client:
                        frame_list.append(self.front_frame)
                    frame_list.append(frame)
            elif self.in_merge:
                has_handle = self.handle_special_key(self.front_frame)
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
                            is_from_client=True
                        )
                        temp_frame_2 = temp_frame.model_copy()
                        temp_frame_2.is_from_client = False
                        frame_list.append(temp_frame)
                        frame_list.append(temp_frame_2)
                    if self.front_frame.is_from_client:
                        frame_list.append(self.front_frame)
                    frame_list.append(frame)
                    self.merge_buffer = []
                    self.merge_buffer_cursor = 0
            self.front_frame = frame
        return [{"timestamp": frame.timestamp, "data": frame.data, "is_from_client": frame.is_from_client} for frame in frame_list]