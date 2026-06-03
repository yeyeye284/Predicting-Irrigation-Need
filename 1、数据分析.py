import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

'''一、导入数据，检查数据基本信息'''
# 1. 导入数据
test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')
sample_submission = pd.read_csv('sample_submission.csv')

# 2. 查看数据集大小
print(f"测试集test 行数={test.shape[0]}，列数={test.shape[1]}")
print(f"训练集train 行数={train.shape[0]}，列数={train.shape[1]}")
print(f"sample_submission 行数={sample_submission.shape[0]}，列数：{sample_submission.shape[1]}\n")

# 3. 查看数据基本信息
print(f"测试集test信息:")
test.info()
print()
print(f"训练集train信息:")
train.info()
print()

# 4. 检查缺失值（结果：没有缺失值）
print(f"训练集train缺失值:{train.isnull().sum().sum()}")
print(f"测试集test缺失值:{test.isnull().sum().sum()}")

'''二、查看训练集中变量分布'''
# 1. 查看目标变量（Irrigation_Need）分布
print("目标变量分布（百分比）:")
print(train['Irrigation_Need'].value_counts(normalize=True))

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.figure(figsize=(10, 6))
sns.countplot(x='Irrigation_Need', data=train)
plt.title('目标变量分布')
plt.xlabel("Irrigation_Need")
plt.ylabel("数量")
plt.savefig("目标变量分布.png",dpi=300,bbox_inches="tight",transparent=False)
# plt.show()

# 2. 查看文本型变量分布特征（画8个图）
categorical_features = train.select_dtypes(include=['object']).columns
categorical_features = [col for col in categorical_features if col != 'Irrigation_Need']

print("分类特征:", categorical_features)
for col in categorical_features:
    print(f"\n{col} 的唯一值数量: {train[col].nunique()}")
    print(f"{col} 的值分布:")
    print(train[col].value_counts().head())

fig, axes = plt.subplots(2, 4, figsize=(19, 10))
axes = axes.flatten()
for i, col in enumerate(categorical_features[:8]):
    sns.countplot(x=col, data=train, ax=axes[i])
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("数量")
plt.tight_layout(pad=2.0)
plt.savefig("文本型变量分布.png",dpi=300,bbox_inches="tight",transparent=False)
# plt.show()

# 3. 查看数值特征分布（画11个图）
numeric_features = train.select_dtypes(include=['int64', 'float64']).columns
numeric_features = [col for col in numeric_features if col not in ['id', 'Irrigation_Need']]
fig, axes = plt.subplots(3, 4, figsize=(15, 10))
axes = axes.flatten()
for i, col in enumerate(numeric_features[:11]):
    sns.histplot(train[col], ax=axes[i], kde=True)
    axes[i].set_title(f'{col}分布')
    axes[i].set_xlabel('')
plt.tight_layout(pad=3.0)
plt.savefig("数值特征分布.png",dpi=300,bbox_inches="tight",transparent=False)
# plt.show()