# Predicting-Irrigation-Need

项目概述
本项目基于Kaggle的"Playground Series - Season 6, Episode 4"竞赛，目标是预测农田的灌溉需求等级（低、中、高）。项目包含完整的数据科学工作流，从探索性数据分析到机器学习模型构建与预测。

# 代码文件说明

1、数据分析.py - 探索性数据分析
功能描述：
此文件包含了完整的数据探索流程，用于理解数据特征、分布和质量，为后续建模提供基础。
运行前准备：
确保train.csv、test.csv和sample_submission.csv文件位于相同目录下

2、机器学习.py - 机器学习模型构建
功能描述：
此文件实现了完整的机器学习流程，从数据预处理到模型训练、评估和预测。

# 运行本代码前需要安装以下Python库：
# 核心数据处理库
pip install pandas>=1.3.0
pip install numpy>=1.21.0

# 数据可视化库
pip install matplotlib>=3.4.0
pip install seaborn>=0.11.0

# 机器学习库
pip install scikit-learn>=1.0.0
pip install xgboost>=1.5.0
pip install lightgbm>=3.3.0
pip install catboost>=1.0.0

# 其他工具
pip install warnings
