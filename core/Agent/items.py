import abc

class RunItem(abc.ABC):
    """一个由代理生成的项。"""

    @abc.abstractmethod
    def to_input_item(self) -> ResponseInputItemParam:
        """将此项转换为适合传递给模型的输入项。"""

