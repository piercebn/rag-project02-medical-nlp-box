from transformers import pipeline
import torch
import logging
import numpy as np
import re
import string

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinNERService:
    """
    金融命名实体识别服务
    """
    def __init__(self):
        # 初始化 NER 模型，使用 GPU 如果可用
        self.pipe = pipeline("token-classification", 
                           model="ckiplab/bert-base-chinese-ws",  # Example model; replace with a financial NER model
                           aggregation_strategy='simple',
                           device=0 if torch.cuda.is_available() else -1)
  
    def process(self, text, term_types):
        """
        处理输入文本，识别金融实体
        
        Args:
            text: 输入文本
            term_types: 需要识别的术语类型
            
        Returns:
            包含识别出的实体和原始文本的字典
        """
        # 使用模型进行实体识别
        result = self.pipe(text)

        # 打印result
        print(result)

        # 确保结果是实体列表
        if isinstance(result, dict):
            result = result.get('entities', [])
        
        # 转换 numpy.float32 为 Python float
        processed_entities = []
        for entity in result:
            processed_entity = {
                'entity_group': entity['entity_group'],
                'word': entity['word'],
                'start': entity['start'],
                'end': entity['end'],
                'score': float(entity['score'])  # 转换 numpy.float32 为 float
            }
            processed_entities.append(processed_entity)
        
        # 合并相邻的 B 和 I 实体
        merged_entities = self._merge_b_i_entities(processed_entities)
        
        # 根据术语类型过滤实体
        filtered_result = self._filter_entities(merged_entities, term_types)
        
        return {
            "text": text,
            "entities": filtered_result
        }

    def _merge_b_i_entities(self, entities):
        """
        合并相邻的 B 和 I 实体
        """
        merged_entities = []
        i = 0
        n = len(entities)
        
        # 中英文标点符号集合（修正字符串定义）
        punctuation_chars = set(string.punctuation + '，。、；："\'[]《》？！…—～')
        
        while i < n:
            current = entities[i]
            if current['entity_group'] == 'B':
                # 初始化合并后的实体
                merged_entity = {
                    'entity_group': 'E',
                    'word': current['word'],
                    'start': current['start'],
                    'end': current['end'],
                    'score': current['score']
                }
                
                # 检查后续的 I 实体
                j = i + 1
                while j < n and entities[j]['entity_group'] == 'I' and entities[j]['start'] == merged_entity['end']:
                    merged_entity['word'] += entities[j]['word']
                    merged_entity['end'] = entities[j]['end']
                    merged_entity['score'] = (merged_entity['score'] + entities[j]['score']) / 2  # 平均得分
                    j += 1
                
                # 去除特殊符号
                merged_entity['word'] = re.sub(r'[#\s，]', '', merged_entity['word'])
                
                # 跳过已合并的 I 实体
                i = j
                
                # 过滤条件：仅过滤纯标点符号的条目
                if not all(char in punctuation_chars for char in merged_entity['word']):
                    merged_entities.append(merged_entity)
            else:
                i += 1
        
        return merged_entities

    def _filter_entities(self, entities, term_types):
        """
        根据术语类型过滤实体
        """
        filtered_result = []
        for entity in entities:
            if term_types.get('allFinancialTerms', False):
                filtered_result.append(entity)
            elif (term_types.get('money', False) and entity['entity_group'] == 'MONEY') or \
                 (term_types.get('revenue', False) and entity['entity_group'] == 'REVENUE') or \
                 (term_types.get('expense', False) and entity['entity_group'] == 'EXPENSE') or \
                 (term_types.get('profit', False) and entity['entity_group'] == 'PROFIT'):
                filtered_result.append(entity)
            else:
                # 默认保留所有其他实体
                filtered_result.append(entity)
        return filtered_result 