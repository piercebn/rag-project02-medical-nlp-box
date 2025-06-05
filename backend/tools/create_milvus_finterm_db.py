from pymilvus import model
from pymilvus import MilvusClient
import pandas as pd
from tqdm import tqdm
import logging
from dotenv import load_dotenv
load_dotenv()
import torch    
from pymilvus import MilvusClient, DataType, FieldSchema, CollectionSchema

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 初始化嵌入函数
embedding_function = model.dense.SentenceTransformerEmbeddingFunction(
    model_name='BAAI/bge-m3',
    device='cuda:0' if torch.cuda.is_available() else 'cpu',
    trust_remote_code=True
)

# 文件路径配置
file_path = "backend/data/FINTERM_ALL.csv"  # 替换为您的金融术语CSV文件路径
db_path = "backend/db/finterm_bge_m3.db"  # 修改为金融术语数据库路径

# 连接到 Milvus
client = MilvusClient(db_path)
collection_name = "concepts_only_name"

# 加载数据
logging.info("Loading financial terms data from CSV")
df = pd.read_csv(file_path, dtype=str, low_memory=False).fillna("NA")

# 获取向量维度
sample_embedding = embedding_function(["Sample Financial Term"])[0]
vector_dim = len(sample_embedding)

# 构造Schema (简化版，只包含必要字段)
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=vector_dim),
    FieldSchema(name="concept_name", dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name="vocabulary_id", dtype=DataType.VARCHAR, max_length=20),
]
schema = CollectionSchema(fields, "Financial Terms Collection", enable_dynamic_field=True)

# 创建集合
if not client.has_collection(collection_name):
    client.create_collection(
        collection_name=collection_name,
        schema=schema
    )
    logging.info(f"Created new collection for financial terms: {collection_name}")

# 创建索引
index_params = client.prepare_index_params()
index_params.add_index(
    field_name="vector",
    index_type="AUTOINDEX",
    metric_type="COSINE",
    params={"nlist": 1024}
)
client.create_index(collection_name=collection_name, index_params=index_params)

# 批量处理数据
batch_size = 1024
for start_idx in tqdm(range(0, len(df), batch_size), desc="Processing financial terms"):
    end_idx = min(start_idx + batch_size, len(df))
    batch_df = df.iloc[start_idx:end_idx]

    # 准备文档 (只使用概念名称)
    docs = [row['concept_name'] for _, row in batch_df.iterrows()]

    # 生成嵌入
    try:
        embeddings = embedding_function(docs)
    except Exception as e:
        logging.error(f"Error generating embeddings: {e}")
        continue

    # 准备数据
    data = [
        {
            "vector": embeddings[idx],
            "concept_name": str(row['concept_name']),
            "vocabulary_id": str(row['vocabulary_id']),
        } for idx, (_, row) in enumerate(batch_df.iterrows())
    ]

    # 插入数据
    try:
        client.insert(collection_name=collection_name, data=data)
    except Exception as e:
        logging.error(f"Error inserting batch: {e}")

logging.info("Financial terms database creation completed.")

# 示例查询
query = "profit"
query_embedding = embedding_function([query])[0].tolist()
search_result = client.search(
    collection_name=collection_name,
    data=[query_embedding],
    limit=5,
    output_fields=["concept_name", "vocabulary_id"]
)
logging.info(f"Search results for '{query}': {search_result}") 