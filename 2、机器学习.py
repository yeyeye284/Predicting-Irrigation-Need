import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import warnings
import numpy as np
warnings.filterwarnings('ignore')

test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')
sample_submission = pd.read_csv('sample_submission.csv')

# 字符串列的分类特征
categorical_features = train.select_dtypes(include=['object']).columns
categorical_features = [col for col in categorical_features if col != 'Irrigation_Need']

'''一、将变量处理成机器学习可以使用的形式'''
# 1. 分离特征和目标变量
X = train.drop(['id', 'Irrigation_Need'], axis=1)
y = train['Irrigation_Need']
test_ids = test['id']
test_features = test.drop('id', axis=1)

# 2. 处理分类变量
label_encoders = {}
for col in categorical_features:
    # 合并训练和测试数据以确保编码一致性
    combined = pd.concat([X[col], test_features[col]], axis=0)
    le = LabelEncoder()
    le.fit(combined)
    X[col] = le.transform(X[col])
    test_features[col] = le.transform(test_features[col])
    label_encoders[col] = le

# 3. 处理目标变量
label_encoder_y = LabelEncoder()
y_encoded = label_encoder_y.fit_transform(y)

'''二、特征工程'''
# 1. 创建交互特征
def create_interaction_features(df):
    df_copy = df.copy()
    # 土壤相关交互特征
    df_copy['soil_moisture_ph'] = df_copy['Soil_Moisture'] * df_copy['Soil_pH']
    df_copy['organic_conductivity'] = df_copy['Organic_Carbon'] * df_copy['Electrical_Conductivity']
    # 气象相关交互特征
    df_copy['temp_humidity'] = df_copy['Temperature_C'] * df_copy['Humidity']
    df_copy['rain_sunlight'] = df_copy['Rainfall_mm'] / (df_copy['Sunlight_Hours'] + 1)
    df_copy['evapotranspiration'] = df_copy['Temperature_C'] / (df_copy['Humidity'] + 1)
    # 面积与灌溉量比值
    df_copy['irrigation_per_area'] = df_copy['Previous_Irrigation_mm'] / (df_copy['Field_Area_hectare'] + 1)

    # 创建多项式特征（平方）
    numeric_cols = ['Soil_pH', 'Soil_Moisture', 'Temperature_C', 'Rainfall_mm', 'Sunlight_Hours']
    for col in numeric_cols:
        if col in df_copy.columns:
            df_copy[f'{col}_squared'] = df_copy[col] ** 2

    return df_copy


# 应用特征工程
X_engineered = create_interaction_features(X)
test_engineered = create_interaction_features(test_features)
print(f"原始特征数量: {X.shape[1]}")
print(f"工程后特征数量: {X_engineered.shape[1]}")


'''三、尝试运行一次XGBoost，并验证'''
# 1. 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(
    X_engineered, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)
print(f"训练集大小: {X_train.shape}")
print(f"验证集大小: {X_val.shape}\n")

# 2. 训练XGBoost模型
xgb_model = xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='mlogloss',
    use_label_encoder=False,
    # early_stopping_rounds=50 ##
)

print("xgb_model.fit()执行：")
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=100
)

# 3. 在验证集、训练集上评估
y_pred_val = xgb_model.predict(X_val)
val_score = balanced_accuracy_score(y_val, y_pred_val)
print(f"XGBoost验证集的平衡准确率: {val_score:.4f}")

y_pred_train = xgb_model.predict(X_train)
train_score = balanced_accuracy_score(y_train, y_pred_train)
print(f"训练集平衡准确率: {train_score:.4f}\n")

# 4. 特征重要性分析
feature_importance = pd.DataFrame({
    'feature': X_engineered.columns,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
plt.title('Top 20特征重要性')
plt.tight_layout()
plt.savefig("Top 20特征重要性.png",dpi=300,bbox_inches="tight",transparent=False)
# plt.show()

print("Top 20 重要特征:")
print(feature_importance.head(20))


'''四、交叉验证'''
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    xgb_model, X_engineered, y_encoded,
    cv=cv, scoring='balanced_accuracy'
)
print(f"5折交叉验证平均分: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

'''五、训练多个模型进行集成，进行模型融合验证'''
# 1. 建立模型
models = {
    'xgb': xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='mlogloss',
        use_label_encoder=False
    ),
    'lgb': lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    ),
    'catboost': CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        random_seed=42,
        verbose=0
    )
}

# 2. 训练所有模型
predictions = {}
for name, model in models.items():
    print(f"\n训练 {name} 模型...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    score = balanced_accuracy_score(y_val, y_pred)
    print(f"{name} 验证集平衡准确率: {score:.4f}")
    predictions[name] = model.predict_proba(X_engineered)

# 3. 模型融合（加权平均）
xgb_weight = 0.4
lgb_weight = 0.3
cat_weight = 0.3

final_predictions_proba = (
    predictions['xgb'] * xgb_weight +
    predictions['lgb'] * lgb_weight +
    predictions['catboost'] * cat_weight
)

final_predictions = np.argmax(final_predictions_proba, axis=1)
ensemble_score = balanced_accuracy_score(y_encoded, final_predictions)
print(f"\n集成模型在完整训练集上的平衡准确率: {ensemble_score:.4f}")


'''六、用训练好的模型预测结果，并输出'''
# 1. 使用最佳模型对测试集进行预测（XGBoost）
final_model = xgb_model

test_predictions_proba = final_model.predict_proba(test_engineered)
test_predictions = np.argmax(test_predictions_proba, axis=1)

# 2. 将编码转换回原始标签
test_predictions_labels = label_encoder_y.inverse_transform(test_predictions)

# 3. 创建提交文件
submission = pd.DataFrame({
    'id': test_ids,
    'Irrigation_Need': test_predictions_labels
})

# 4. 保存为CSV文件
submission.to_csv('submission.csv', index=False)
print("submission.csv已保存")