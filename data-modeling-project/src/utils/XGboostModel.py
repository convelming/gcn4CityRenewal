import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import matplotlib.pyplot as plt

def calculate_cpc(y_true, y_pred):
    """计算CPC指标"""
    min_sum = np.sum(np.minimum(y_true, y_pred))
    total_sum = np.sum(y_true) + np.sum(y_pred)
    cpc = (2 * min_sum) / total_sum if total_sum != 0 else 0
    return cpc

def prepare_data(df):
    """准备数据，将input_x从字符串转换为数组"""
    df = df.copy()
    df['input_x'] = df['input_x'].apply(ast.literal_eval)
    X = np.array(df['input_x'].tolist())
    y = df['flowCounts'].values
    return X, y

def train_xgboost(df, test_size=0.2, random_state=42, params=None):
    """
    训练XGBoost模型并输出训练过程的损失函数
    
    参数:
        df: 包含o_id, input_x, flowCounts的DataFrame
        test_size: 测试集比例
        random_state: 随机种子
        params: XGBoost参数字典
        
    返回:
        model: 训练好的模型
        X_test: 测试集特征
        y_test: 测试集真实值
        eval_results: 训练评估结果
    """
    # 准备数据
    X, y = prepare_data(df)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # 默认参数
    if params is None:
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'eta': 0.1,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'seed': random_state,
            'nthread': -1
        }
    
    # 创建DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # 训练模型
    eval_results = {}
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=3000,
        evals=[(dtrain, 'train'), (dtest, 'test')],
        early_stopping_rounds=500,
        evals_result=eval_results,
        verbose_eval=False
    )
    
#     # 绘制训练过程
#     plt.figure(figsize=(10, 6))
#     plt.plot(eval_results['train']['rmse'], label='Train RMSE')
#     plt.plot(eval_results['test']['rmse'], label='Test RMSE')
#     plt.xlabel('Boosting Rounds')
#     plt.ylabel('RMSE')
#     plt.title('Training and Validation Loss')
#     plt.legend()
#     plt.grid()
#     plt.show()
    
    # 评估模型
#     y_pred = model.predict(dtest)
#     mse = mean_squared_error(y_test, y_pred)
#     mae = mean_absolute_error(y_test, y_pred)
#     rmse = np.sqrt(mse)
#     cpc = calculate_cpc(y_test, y_pred)
    
#     print(f"\nModel Evaluation:")
#     print(f"Test RMSE: {rmse:.4f}")
#     print(f"Test MAE: {mae:.4f}")
#     print(f"Test MSE: {mse:.4f}")
#     print(f"Test CPC: {cpc:.4f}")
    
    return model, X_test, y_test, eval_results

def predict_with_model(model, new_data):
    """
    使用训练好的模型进行预测
    
    参数:
        model: 训练好的XGBoost模型
        new_data: 新数据DataFrame，包含input_x列
        
    返回:
        predictions: 预测结果数组
    """
    if isinstance(new_data, pd.DataFrame):
        new_data['input_x'] = new_data['input_x'].apply(ast.literal_eval)
        X_new = np.array(new_data['input_x'].tolist())
    else:
        X_new = np.array(new_data)
    
    dnew = xgb.DMatrix(X_new)
    predictions = model.predict(dnew)
    return predictions

import math

def calculate_std_dev(data):
    n = len(data)
    mean = sum(data) / n
    squared_diff = [(x - mean) ** 2 for x in data]
    variance = sum(squared_diff) / n  # 总体方差
    std_dev = math.sqrt(variance)     # 标准差
    return std_dev

def calculate_rmse(y_true, y_pred):
    n = len(y_true)
    squared_errors = [(true - pred) ** 2 for true, pred in zip(y_true, y_pred)]
    mse = sum(squared_errors) / n  # 均方误差（MSE）
    rmse = math.sqrt(mse)          # 开平方根
    return rmse
    
def fu_to_zero(value_data):
    if value_data<0:
        return(0)
    else:
        return(value_data)    