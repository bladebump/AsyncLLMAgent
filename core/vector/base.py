from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, TypeVar, Generic
from dataclasses import dataclass, field

T = TypeVar('T')


class DocumentBase:
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


@dataclass
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
    sparse_vector: List[float]
    filename: str
    department: int
    
    def to_insert_data(self) -> Dict[str, Any]:
        """
        将文档转换为插入数据，用于插入到向量数据库中
        """
        return {
            "text": self.text,
            "dense_vector": self.dense_vector,
            "sparse_vector": self.sparse_vector,
            "filename": self.filename,
            "department": self.department,
        }

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
        sparse_vector = data.pop("sparse_vector", [])
        filename = data.pop("filename", "")
        department = data.pop("department", "")
        # 构建Document对象
        return cls(
            text=text, 
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            filename=filename,
            department=department
        )


class VectorStoreBase(Generic[T], ABC):
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
            如果文档ID由用户提供，成功返回True，否则返回False
            如果文档ID自动生成，成功返回生成的ID，失败返回False
        """
        pass
    
    @abstractmethod
    async def add_batch(self, documents: List[Document], collection_name: str) -> int:
        """
        批量添加文档
        
        参数:
            documents: 要添加的文档对象列表
            collection_name: 集合/索引名称，如果未提供则使用默认值
            
        返回:
            如果所有文档ID都由用户提供，成功返回True，否则返回False
            如果存在自动生成的ID，成功返回生成的ID列表，失败返回False
        """
        pass
    
    @abstractmethod
    async def get(self, id: str, collection_name: str) -> Optional[Document]:
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
    async def update(self, document: Document, collection_name: str) -> bool:
        """
        更新文档
        
        参数:
            document: 要更新的文档对象，必须包含id
            collection_name: 集合/索引名称，如果未提供则使用默认值
            
        返回:
            成功返回True，否则返回False
        """
        pass
    
    @abstractmethod
    async def delete(self, id: str, collection_name: str) -> bool:
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
    async def delete_batch(self, ids: List[str], collection_name: str) -> bool:
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
        query_vector: List[float], 
        limit: int = 10, 
        offset: int = 0,
        filter: Optional[Dict[str, Any]] = None,
        collection_name: str = None
    ) -> List[Document]:
        """
        向量相似度搜索
        
        参数:
            query_vector: 查询向量
            limit: 返回结果数量上限
            offset: 结果偏移量，用于分页
            filter: 元数据过滤条件
            collection_name: 集合/索引名称，如果未提供则使用默认值
            
        返回:
            匹配的文档列表，按相似度降序排列
        """
        pass
    
    @abstractmethod
    async def keyword_search(
        self, 
        query: str, 
        fields: List[str], 
        limit: int = 10,
        offset: int = 0,
        filter: Optional[Dict[str, Any]] = None,
        collection_name: str = None
    ) -> List[Document]:
        """
        关键词搜索
        
        参数:
            query: 查询字符串
            fields: 要搜索的字段列表
            limit: 返回结果数量上限
            offset: 结果偏移量，用于分页
            filter: 元数据过滤条件
            collection_name: 集合/索引名称，如果未提供则使用默认值
            
        返回:
            匹配的文档列表，按相关性降序排列
        """
        pass
    
    @abstractmethod
    async def hybrid_search(
        self,
        query_text: str,
        query_vector: List[float],
        fields: List[str],
        limit: int = 10,
        offset: int = 0,
        filter: Optional[Dict[str, Any]] = None,
        alpha: float = 0.5,
        collection_name: str = None
    ) -> List[Document]:
        """
        混合搜索（关键词 + 向量）
        
        参数:
            query_text: 查询文本
            query_vector: 查询向量
            fields: 要搜索的字段列表
            limit: 返回结果数量上限
            offset: 结果偏移量，用于分页
            filter: 元数据过滤条件
            alpha: 混合权重，0表示只使用关键词搜索，1表示只使用向量搜索
            collection_name: 集合/索引名称，如果未提供则使用默认值
            
        返回:
            匹配的文档列表，按混合得分降序排列
        """
        pass 