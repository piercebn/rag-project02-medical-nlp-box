import React, { useState } from 'react';
import { AlertCircle } from 'lucide-react';
import { TextInput } from '../components/shared/ModelOptions';

const color_map = {
  'MONEY': "#FF9800", // Orange
  'REVENUE': "#4CAF50", // Green
  'EXPENSE': "#F44336", // Red
  'PROFIT': "#2196F3", // Blue
  'E': "#FF9800", // Orange (for merged entities)
};

const FinNerPage = () => {
  const [input, setInput] = useState('');
  const [result, setResult] = useState('');
  const [coloredResult, setColoredResult] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [termTypes, setTermTypes] = useState({
    allFinancialTerms: true,
    currency: false,
    income: false,
    expense: false,
    profit: false,
  });

  const handleTermTypeChange = (e) => {
    const { name, checked } = e.target;
    if (name === 'allFinancialTerms') {
      setTermTypes({
        currency: false,
        income: false,
        expense: false,
        profit: false,
        allFinancialTerms: checked,
      });
    } else {
      setTermTypes({ ...termTypes, [name]: checked });
    }
  };

  const handleSubmit = async () => {
    setIsLoading(true);
    try {
      const response = await fetch('http://localhost:8000/api/fin-ner', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: input, termTypes }),
      });
      const data = await response.json();
      setResult(JSON.stringify(data, null, 2));
      setColoredResult(generateColoredResult(data.text, data.entities));
    } catch (error) {
      console.error('Error:', error);
      setResult('An error occurred while processing the request.');
      setColoredResult('');
    }
    setIsLoading(false);
  };

  const generateColoredResult = (text, entities) => {
    let result = text;
    entities.sort((a, b) => b.start - a.start);
    
    for (const entity of entities) {
      const color = color_map[entity.entity_group] || '#000000';
      const highlightedEntity = `<span style="background-color: ${color}; padding: 2px; border-radius: 3px;">
        ${entity.word}<sub>${entity.entity_group}</sub>
      </span>`;
      result = result.slice(0, entity.start) + highlightedEntity + result.slice(entity.end);
    }
    return result;
  };

  return (
    <div className="max-w-2xl mx-auto">
      <h1 className="text-3xl font-bold mb-6">金融命名实体识别 💰</h1>
      <div className="bg-white shadow-md rounded-lg p-6 mb-6">
        <h2 className="text-xl font-semibold mb-4">输入金融文本</h2>
        <TextInput
          value={input}
          onChange={(e) => setInput(e.target.value)}
          rows={4}
          placeholder="请输入需要进行命名实体识别的金融文本..."
        />
        
        <h3 className="text-lg font-semibold mb-2">金融术语类型</h3>
        <div className="mb-4">
          <label className="inline-flex items-center mr-4">
            <input
              type="checkbox"
              name="allFinancialTerms"
              checked={termTypes.allFinancialTerms}
              onChange={handleTermTypeChange}
              className="mr-2"
            />
            所有金融术语
          </label>
          <label className="inline-flex items-center mr-4">
            <input
              type="checkbox"
              name="currency"
              checked={termTypes.currency}
              onChange={handleTermTypeChange}
              className="mr-2"
            />
            货币
          </label>
          <label className="inline-flex items-center mr-4">
            <input
              type="checkbox"
              name="income"
              checked={termTypes.income}
              onChange={handleTermTypeChange}
              className="mr-2"
            />
            收入
          </label>
          <label className="inline-flex items-center mr-4">
            <input
              type="checkbox"
              name="expense"
              checked={termTypes.expense}
              onChange={handleTermTypeChange}
              className="mr-2"
            />
            支出
          </label>
          <label className="inline-flex items-center">
            <input
              type="checkbox"
              name="profit"
              checked={termTypes.profit}
              onChange={handleTermTypeChange}
              className="mr-2"
            />
            利润
          </label>
        </div>

        <button
          onClick={handleSubmit}
          disabled={isLoading}
          className={`bg-blue-500 text-white px-4 py-2 rounded-md hover:bg-blue-600 ${isLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
        >
          {isLoading ? '处理中...' : '识别实体'}
        </button>
      </div>
      {coloredResult && (
        <div className="bg-white shadow-md rounded-lg p-6 mb-6">
          <h2 className="text-xl font-semibold mb-4">识别结果</h2>
          <div 
            dangerouslySetInnerHTML={{ __html: coloredResult }} 
            style={{
              lineHeight: '2',
              wordBreak: 'break-word'
            }}
          />
        </div>
      )}
      {result && (
        <div className="bg-green-100 border-l-4 border-green-500 text-green-700 p-4 mb-6" role="alert">
          <p className="font-bold">JSON 结果：</p>
          <pre>{result}</pre>
        </div>
      )}
      <div className="flex items-center text-yellow-700 bg-yellow-100 p-4 rounded-md">
        <AlertCircle className="mr-2" />
        <span>这是演示版本, 并非所有功能都可以正常工作。更多功能需要您来增强并实现。</span>
      </div>
    </div>
  );
};

export default FinNerPage; 