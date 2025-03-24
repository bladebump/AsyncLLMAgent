from typing import Any, Dict, List, Optional, Union
from pymilvus import DataType, AsyncMilvusClient, MilvusClient, AnnSearchRequest, WeightedRanker
from .base import VectorStoreBase, Document
from utils.log import logger


class MilvusVectorStore(VectorStoreBase):
    """
    基于Milvus的向量存储实现
    
    支持向量搜索、关键词搜索和混合搜索
    使用Milvus的多向量字段功能和混合搜索
    """
    
    def __init__(
        self,
        uri: str = "http://localhost:19530",
        username: str = "",
        password: str = "",
        dense_vector_dim: int = 768,
        use_sparse_vector: bool = True,
        **kwargs
    ):
        """
        初始化Milvus向量存储
        
        参数:
            uri: Milvus服务器地址
            username: 认证用户名
            password: 认证密码
            dense_vector_dim: 密集向量维度
            use_sparse_vector: 是否使用稀疏向量字段
            **kwargs: 其他Milvus客户端参数
        """
        self.uri = uri
        self.username = username
        self.password = password
        self.dense_vector_dim = dense_vector_dim
        self.use_sparse_vector = use_sparse_vector
        
        # 构建连接参数
        conn_params = {"uri": uri}
        if username and password:
            conn_params["user"] = username
            conn_params["password"] = password
        # 创建Milvus客户端
        self.client = MilvusClient(uri=uri,user=username,password=password)
        self.Asyclient = AsyncMilvusClient(uri=uri,user=username,password=password)
        # 从Document类自动生成集合结构
        self.collection_schema = self._generate_collection_schema()
    
    def _generate_collection_schema(self):
        """
        根据Document类生成Milvus集合结构
        
        参数:
            doc_class: 文档类
            
        返回:
            Milvus集合结构配置
        """
        schema = self.client.create_schema(
            auto_id=True,
            description="Document集合结构配置",
        )
        schema.add_field('id',DataType.INT64,is_primary=True)
        schema.add_field('text',DataType.VARCHAR,max_length=4000)
        schema.add_field('filename',DataType.VARCHAR,max_length=512)
        schema.add_field('department',DataType.INT64)
        schema.add_field('dense_vector',DataType.FLOAT_VECTOR,dim=self.dense_vector_dim)
        # 只有在启用稀疏向量时才添加sparse_vector字段
        if self.use_sparse_vector:
            schema.add_field('sparse_vector',DataType.SPARSE_FLOAT_VECTOR)
        return schema
    
    async def _generate_index_params(self):
        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="dense_vector",index_type="AUTOINDEX", metric_type="IP")
        # 只有在启用稀疏向量时才为sparse_vector字段创建索引
        if self.use_sparse_vector:
            index_params.add_index(field_name="sparse_vector",index_type="AUTOINDEX", metric_type="IP")
        index_params.add_index(field_name="text",index_type="AUTOINDEX")
        return index_params
    
    async def _check_collection_exists(self, collection_name: str) -> bool:
        return self.client.has_collection(collection_name)

    async def initialize(self, collection_name: str) -> bool:
        """
        初始化集合
        此方法仅检查集合是否存在并加载到内存，但不会自动创建集合
        集合创建应通过create_collection方法显式执行
        参数:
            collection_name: 要初始化的集合名称，必须提供
        返回:
            初始化成功返回True，失败返回False
        """
        # 检查集合是否存在
        has_collection = await self._check_collection_exists(collection_name)
            
        if not has_collection:
            logger.warning(f"集合 {collection_name} 不存在，请先调用create_collection方法创建")
            return False
            
        # 加载集合到内存
        await self.Asyclient.load_collection(collection_name=collection_name)
        return True
    
    async def create_collection(self, collection_name: str) -> bool:
        """
        创建集合
        显式创建一个新的集合，如果集合已存在则返回False
        参数:
            collection_name: 要创建的集合名称，必须提供
        返回:
            创建成功返回True，失败或已存在返回False
        """
        has_collection = await self._check_collection_exists(collection_name)
            
        if has_collection:
            logger.warning(f"集合 {collection_name} 已存在，无需创建")
            return False
            
        # 创建集合
        await self.Asyclient.create_collection(
            collection_name=collection_name,
            schema=self.collection_schema
        )
        
        if await self._check_collection_exists(collection_name):
            index_params = await self._generate_index_params()
            await self.Asyclient.create_index(collection_name=collection_name,index_params=index_params)
            return True
        else:
            return False
    
    async def drop_collection(self, collection_name: str):
        """
        删除集合
        参数:
            collection_name: 要删除的集合名称
        """
        await self.Asyclient.drop_collection(collection_name=collection_name)
    
    async def get_all_collections(self) -> List[str]:
        """
        获取所有集合名称
        
        返回:
            List[str]: 所有集合名称列表
        """
        return self.client.list_collections()
    
    async def add(self, document: Document, collection_name: str) -> int:
        """
        添加单个文档
        参数:
            document: 要添加的文档对象
            collection_name: 目标集合名称，必须提供
        返回:
            如果文档ID由用户提供，成功返回True，否则返回False
            如果文档ID自动生成，成功返回生成的ID，失败返回False
        """
        if not await self._check_collection_exists(collection_name):
            raise ValueError(f"集合 {collection_name} 不存在，请先调用create_collection方法创建")
        
        data = document.to_insert_data()
        result = await self.Asyclient.insert(collection_name=collection_name,data=data)
        return result['insert_count']
    
    async def add_batch(self, documents: List[Document], collection_name: str) -> Union[bool, List[str]]:
        """
        批量添加文档
        参数:
            documents: 要添加的文档对象列表
            collection_name: 目标集合名称，必须提供
        返回:
            如果所有文档ID都由用户提供，成功返回True，否则返回False
            如果存在自动生成的ID，成功返回生成的ID列表，失败返回False
        """
        # 检查集合是否存在
        if not await self._check_collection_exists(collection_name):
            raise ValueError(f"集合 {collection_name} 不存在，请先调用create_collection方法创建")
        
        data = [document.to_insert_data() for document in documents]
        result = await self.Asyclient.insert(collection_name=collection_name,data=data)
        return result['insert_count']
    
    async def get(self, id: int, collection_name: str) -> Optional[Document]:
        """
        获取指定ID的文档
        参数:
            id: 文档唯一标识
            collection_name: 目标集合名称，必须提供
        返回:
            文档对象，如果不存在返回None
        """
        if not await self._check_collection_exists(collection_name):
            raise ValueError(f"集合 {collection_name} 不存在，请先调用create_collection方法创建")
        
        result = await self.Asyclient.get(collection_name=collection_name,ids=[id])
        return Document.from_dict(result['entities'][0])
    
    async def update(self, id: int, document: Document, collection_name: str) -> bool:
        """
        更新指定ID的文档
        
        参数:
            id: 文档唯一标识
            document: 要更新的文档对象
            collection_name: 目标集合名称，必须提供
        返回:
            成功返回True，否则返回False
        """
        if not await self._check_collection_exists(collection_name):
            raise ValueError(f"集合 {collection_name} 不存在，请先调用create_collection方法创建")
        
        result = await self.delete(id,collection_name)
        if result:
            return await self.add(document,collection_name)
        return False
    
    async def delete(self, id: str, collection_name: str) -> int:
        """
        删除指定ID的文档
        
        参数:
            id: 文档唯一标识
            collection_name: 目标集合名称，必须提供
            
        返回:
            成功返回True，否则返回False
        """
        if not await self._check_collection_exists(collection_name):
            raise ValueError(f"集合 {collection_name} 不存在，请先调用create_collection方法创建")
        
        result = await self.Asyclient.delete(collection_name=collection_name,ids=[id])
        return result['delete_count']
    
    async def delete_batch(self, ids: List[str], collection_name: str) -> int:
        """
        批量删除文档
        
        参数:
            ids: 文档唯一标识列表
            collection_name: 目标集合名称，必须提供
            
        返回:
            成功返回True，否则返回False
        """
        if not await self._check_collection_exists(collection_name):
            raise ValueError(f"集合 {collection_name} 不存在，请先调用create_collection方法创建")
        
        result = await self.Asyclient.delete(collection_name=collection_name,ids=ids)
        return result['delete_count']
    
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
        if not await self.initialize(collection_name):
            raise ValueError(f"集合 {collection_name} 初始化失败，请检查集合是否存在")
            
        # 如果是sparse_vector字段且未启用稀疏向量
        if anns_field == "sparse_vector" and not self.use_sparse_vector:
            raise ValueError("稀疏向量字段未启用，请使用dense_vector字段")

        search_params = {
            "metric_type": "IP",
            "params": {"nprobe": 10}
        }
        result = await self.Asyclient.search(
            collection_name=collection_name,
            data=dense_vector,
            anns_field=anns_field,
            limit=limit,
            filter=filter,
            output_fields=output_fields,
            search_params=search_params
        )
        return [Document.from_dict(hit['entity']) for hits in result for hit in hits]
    
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
        if not self.use_sparse_vector and anns_field == "sparse_vector":
            raise ValueError("稀疏向量字段未启用，请使用dense_vector或其他字段")
            
        return await self.vector_search(sparse_vector,collection_name,anns_field,output_fields,limit,filter)
        
    
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
        if not await self._check_collection_exists(collection_name):
            raise ValueError(f"集合 {collection_name} 不存在，请先调用create_collection方法创建")
        
        if not await self.initialize(collection_name):
            raise ValueError(f"集合 {collection_name} 初始化失败，请检查集合是否存在")
        
        # 如果不使用稀疏向量或未提供稀疏向量，则只进行密集向量搜索
        if not self.use_sparse_vector or sparse_vector is None:
            return await self.vector_search(
                dense_vector=dense_vector,
                collection_name=collection_name,
                anns_field="dense_vector",
                output_fields=output_fields,
                limit=limit,
                filter=filter
            )
        
        dense_search_params = {'metric_type': 'IP'}
        sparse_search_params = {'metric_type': 'IP'}
        dense_search_request = AnnSearchRequest(
            data=dense_vector,
            anns_field="dense_vector",
            param=dense_search_params,
            limit=limit,
            expr=filter
        )
        sparse_search_request = AnnSearchRequest(
            data=sparse_vector,
            anns_field="sparse_vector",
            param=sparse_search_params,
            limit=limit,
            expr=filter
        )
        result = await self.Asyclient.hybrid_search(
            collection_name=collection_name,
            reqs=[dense_search_request, sparse_search_request],
            ranker=WeightedRanker(dense_weight,sparse_weight),
            output_fields=output_fields,
            limit=limit,
        )
        return [Document.from_dict(hit['entity']) for hits in result for hit in hits]

    async def release_collection(self, collection_name: str) -> bool:
        """
        从内存中释放集合
        
        参数:
            collection_name: 要释放的集合名称
            
        返回:
            释放成功返回True，失败返回False
        """
        if not await self._check_collection_exists(collection_name):
            logger.warning(f"集合 {collection_name} 不存在，无需释放")
            return False
        
        await self.Asyclient.release_collection(collection_name=collection_name)
        logger.info(f"集合 {collection_name} 已从内存中释放")
        return True

    async def close(self):
        """
        关闭连接，释放所有集合资源
        """
        # 获取所有集合
        collections = self.client.list_collections()
        
        # 释放所有集合
        for collection_name in collections:
            # 尝试释放每个集合，但继续处理其他集合
            try:
                await self.release_collection(collection_name)
            except Exception as e:
                # 这里仍捕获异常是为了确保其他集合能被释放
                logger.error(f"释放集合 {collection_name} 失败: {str(e)}")
        
        logger.info("所有集合已释放，资源已清理") 