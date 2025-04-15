from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypeVar, Generic, Union
from pydantic import BaseModel
T = TypeVar('T')

class DocumentBase(BaseModel):
    """
    向量文档基类，所有文档类型应继承此类
    """
    
    def to_insert_data(self) -> Dict[str, Any]:
        """
        将文档转换为插入数据，用于插入到向量数据库中
        子类可根据需要重写此方法
        """
        raise NotImplementedError("子类必须实现to_insert_data方法")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将文档转换为字典表示，用于存储
        子类可根据需要重写此方法
        """
        raise NotImplementedError("子类必须实现to_dict方法")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentBase':
        """
        从存储的字典数据构建Document对象
        子类可根据需要重写此方法
        """
        raise NotImplementedError("子类必须实现from_dict方法")

class Document(DocumentBase):
    """
    向量文档基类，表示一个存储在向量数据库中的文档
    
    属性:
        text: 文档文本内容
        vector: 文档的向量表示
        filename: 文件名
        department: 所属部门
        id: 文档唯一标识，可选，如未提供则由向量存储自动生成
        metadata: 其他元数据信息，以键值对形式存储
    """
    text: str
    dense_vector: List[float]
    sparse_vector: Optional[List[float]] = None
    filename: str
    department: int
    
    def to_insert_data(self) -> Dict[str, Any]:
        """
        将文档转换为插入数据，用于插入到向量数据库中
        """
        data = {
            "text": self.text,
            "dense_vector": self.dense_vector,
            "filename": self.filename,
            "department": self.department,
        }
        if self.sparse_vector:
            data["sparse_vector"] = self.sparse_vector
        return data

    def to_dict(self) -> Dict[str, Any]:
        """
        将文档转换为字典表示，用于存储
        """
        return {
            "text": self.text,
            "filename": self.filename,
            "department": self.department,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        """
        从存储的字典数据构建Document对象
        
        参数:
            doc_id: 文档ID
            vector: 文档向量
            metadata: 元数据字典，应包含text、filename、department和其他元数据
        """
        # 提取特定字段
        text = data.pop("text", "")
        dense_vector = data.pop("dense_vector", [])
        sparse_vector = data.pop("sparse_vector", None)
        filename = data.pop("filename", "")
        department = data.pop("department", 0)
        # 构建Document对象
        return cls(
            text=text, 
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            filename=filename,
            department=department
        )


class VectorStoreBase(ABC):
    """
    向量存储的基类，定义了向量数据库的基本操作接口
    
    所有的向量存储实现都应该继承这个基类，并实现其抽象方法
    """
    
    @abstractmethod
    async def add(self, document: Document, collection_name: str) -> int:
        """
        添加单个文档
        
        参数:
            document: 要添加的文档对象
            collection_name: 集合/索引名称，如果未提供则使用默认值
            
        返回:
            成功插入的文档数量
        """
        pass
    
    @abstractmethod
    async def add_batch(self, documents: List[Document], collection_name: str, batch_size: int = 1000) -> int:
        """
        批量添加文档
        
        参数:
            documents: 要添加的文档对象列表
            collection_name: 集合/索引名称，如果未提供则使用默认值
            batch_size: 批量大小，如果未提供则使用默认值
        返回:
            成功插入的文档数量
        """
        pass
    
    @abstractmethod
    async def get(self, id: int, collection_name: str) -> Optional[Document]:
        """
        获取指定ID的文档
        
        参数:
            id: 文档唯一标识
            collection_name: 集合/索引名称，如果未提供则使用默认值
            
        返回:
            文档对象，如果不存在返回None
        """
        pass
    
    @abstractmethod
    async def update(self, id: int, document: Document, collection_name: str) -> bool:
        """
        更新文档
        
        参数:
            id: 文档唯一标识
            document: 要更新的文档对象，必须包含id
            collection_name: 集合/索引名称，如果未提供则使用默认值
            
        返回:
            成功返回True，否则返回False
        """
        pass
    
    @abstractmethod
    async def delete(self, id: str, collection_name: str) -> int:
        """
        删除指定ID的文档
        
        参数:
            id: 文档唯一标识
            collection_name: 集合/索引名称，如果未提供则使用默认值
            
        返回:
            成功返回True，否则返回False
        """
        pass
    
    @abstractmethod
    async def delete_batch(self, ids: List[str], collection_name: str) -> int:
        """
        批量删除文档
        
        参数:
            ids: 文档唯一标识列表
            collection_name: 集合/索引名称，如果未提供则使用默认值
            
        返回:
            成功返回True，否则返回False
        """
        pass
    
    @abstractmethod
    async def vector_search(
        self, 
        dense_vector: List[List[float]],
        collection_name: str, 
        anns_field: str,
        output_fields: List[str] = ["text", "filename", "department"],
        limit: int = 10, 
        filter: str = ""
    ) -> List[Document]:
        """
        向量相似度搜索
        
        参数:
            dense_vector: 查询向量
            collection_name: 目标集合名称，必须提供
            anns_field: 向量字段名，必须提供
            output_fields: 返回字段列表
            limit: 返回结果数量上限
            filter: 元数据过滤条件
            
        返回:
            匹配的文档列表，按相似度降序排列
        """
        pass
    
    @abstractmethod
    async def keyword_search(
        self, 
        sparse_vector: List[List[float]],
        collection_name: str, 
        anns_field: str = "sparse_vector",
        output_fields: List[str] = ["text", "filename", "department"],
        limit: int = 10, 
        filter: str = ""
    ) -> List[Document]:
        """
        关键词搜索
        
        参数:
            sparse_vector: 查询向量
            collection_name: 目标集合名称，必须提供
            anns_field: 向量字段名，默认为sparse_vector
            output_fields: 返回字段列表
            limit: 返回结果数量上限
            filter: 元数据过滤条件
            
        返回:
            匹配的文档列表，按相关性降序排列
        """
        pass
    
    @abstractmethod
    async def hybrid_search(
        self,
        dense_vector: List[List[float]],
        sparse_vector: List[List[float]] = None,
        collection_name: str = None, 
        output_fields: List[str] = ["text", "filename", "department"],
        limit: int = 10, 
        filter: str = "",
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0
    ) -> List[Document]:
        """
        混合搜索（关键词 + 向量）
        
        参数:
            dense_vector: 查询向量
            sparse_vector: 查询稀疏向量（可选，当use_sparse_vector=False时可以为None）
            collection_name: 目标集合名称，必须提供
            output_fields: 返回字段列表
            limit: 返回结果数量上限
            filter: 元数据过滤条件
            dense_weight: 密集向量权重
            sparse_weight: 稀疏向量权重
            
        返回:
            匹配的文档列表，按混合得分降序排列
        """
        pass 