import optuna
import numpy as np
from sklearn.metrics import accuracy_score  # 或你选择的其他评估指标


def objective(trial):
    
    # 待优化的权重
    w1 = trial.suggest_float("w1", 0, 1)
    w2 = trial.suggest_float("w2", 0, 1)
    w3 = trial.suggest_float("w3", 0, 1)
    w4 = trial.suggest_float("w4", 0, 1)
    w5 = 1 - (w1 + w2 + w3 + w4)  # 确保权重之和为1

    # 载入五个模型的预测数据
    preds1 = np.load('./submit/preds.npy') 
    preds2 = np.load('./submit/preds.npy')
    preds3 = np.load('./submit/preds.npy')
    preds4 = np.load('./submit/preds.npy')
    preds5 = np.load('./submit/preds.npy')

    # 预测结果的加权平均
    weighted_preds = (w1 * preds1 + w2 * preds2 + w3 *
                      preds3 + w4 * preds4 + w5 * preds5) / 5

    # 评估模型（你需要定义评估指标和真实标签）
    # 例如，如果使用准确率：
    accuracy = accuracy_score(np.argmax(weighted_preds, axis=1))

    # 这里我们返回一个虚拟值作为示例。请用你实际的评估指标替换这里。
    return accuracy  # 替换为实际的评估结果


# 创建一个研究对象并优化目标函数
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)  # 可以调整试验次数

# 发现的最佳权重
best_weights = study.best_params

