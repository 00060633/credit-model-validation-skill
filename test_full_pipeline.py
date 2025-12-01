#!/usr/bin/env python3
"""
Полный тест-пайплайн для валидации кредитной модели
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def create_synthetic_credit_data(n_samples=10000):
    """Создание синтетических кредитных данных"""
    print("🏗️ Создание синтетических кредитных данных...")
    
    np.random.seed(42)
    
    # Создаем признаки, типичные для кредитного скоринга
    data = {
        'age': np.random.normal(40, 12, n_samples).clip(18, 80),
        'income': np.random.lognormal(10.5, 0.8, n_samples).clip(20000, 500000),
        'credit_history_months': np.random.gamma(2, 20, n_samples).clip(0, 300),
        'existing_loans': np.random.poisson(1.5, n_samples).clip(0, 10),
        'debt_to_income': np.random.beta(2, 5, n_samples) * 100,
        'employment_status': np.random.choice([0, 1, 2], n_samples, p=[0.1, 0.7, 0.2]),  # 0=безработный, 1=работает, 2=пенсионер
        'property_value': np.random.lognormal(12, 0.6, n_samples).clip(50000, 2000000),
        'education_level': np.random.choice([0, 1, 2, 3], n_samples, p=[0.3, 0.4, 0.2, 0.1]),  # 0-3 уровни образования
        'marital_status': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),  # 0=холост, 1=женат
        'region_risk': np.random.choice([0, 1, 2], n_samples, p=[0.6, 0.3, 0.1])  # 0=низкий, 1=средний, 2=высокий риск региона
    }
    
    df = pd.DataFrame(data)
    
    # Создаем целевую переменную (дефолт) на основе признаков
    # Логика: молодые с низким доходом и высоким DTI имеют больший риск дефолта
    default_probability = (
        0.02 +  # базовый риск
        0.03 * (df['age'] < 25).astype(int) +  # молодые
        0.05 * (df['income'] < 50000).astype(int) +  # низкий доход
        0.04 * (df['debt_to_income'] > 40).astype(int) +  # высокий DTI
        0.03 * (df['employment_status'] == 0).astype(int) +  # безработные
        0.02 * (df['existing_loans'] > 3).astype(int) +  # много кредитов
        0.015 * (df['credit_history_months'] < 12).astype(int) +  # короткая кред. история
        0.01 * (df['region_risk'] == 2).astype(int)  # рискованный регион
    )
    
    df['target'] = np.random.binomial(1, default_probability.clip(0, 0.3), n_samples)
    
    print(f"✅ Создано {n_samples} записей")
    print(f"   Доля дефолтов: {df['target'].mean():.3%}")
    
    return df

def create_test_model(X_train, y_train):
    """Создание и обучение тестовой модели"""
    print("🤖 Обучение тестовой модели...")
    
    # Создаем простую модель Random Forest
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    
    print("✅ Модель обучена")
    return model

def main():
    """Основная функция для создания тестовых данных"""
    
    # Создаем структуру проекта
    print("📁 Создание структуры проекта...")
    os.makedirs("test_data", exist_ok=True)
    os.makedirs("test_models", exist_ok=True)
    
    # Создаем синтетические данные
    df = create_synthetic_credit_data(10000)
    
    # Разделяем на train/validation для симуляции реального процесса
    features = [col for col in df.columns if col != 'target']
    X = df[features]
    y = df['target']
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Создаем и сохраняем модель
    model = create_test_model(X_train, y_train)
    
    model_path = "test_models/credit_model_test.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"💾 Модель сохранена: {model_path}")
    
    # Сохраняем валидационные данные
    val_data = pd.concat([X_val, y_val], axis=1)
    val_path = "test_data/validation_data.csv"
    val_data.to_csv(val_path, index=False)
    print(f"💾 Валидационные данные сохранены: {val_path}")
    
    # Сохраняем тренировочные данные для PSI анализа
    train_data = pd.concat([X_train, y_train], axis=1)
    train_path = "test_data/train_data.csv"
    train_data.to_csv(train_path, index=False)
    print(f"💾 Тренировочные данные сохранены: {train_path}")
    
    # Выводим базовую статистику
    print("\n📊 Статистика тестовых данных:")
    print(f"  Общий размер выборки: {len(df)}")
    print(f"  Тренировочная выборка: {len(X_train)}")
    print(f"  Валидационная выборка: {len(X_val)}")
    print(f"  Количество признаков: {len(features)}")
    print(f"  Доля дефолтов (train): {y_train.mean():.3%}")
    print(f"  Доля дефолтов (validation): {y_val.mean():.3%}")
    
    print("\n🎯 Тестовые данные готовы для валидации!")
    print("\nСледующие шаги:")
    print("1. Запустить calculate_metrics.py")
    print("2. Запустить stability_analysis.py") 
    print("3. Запустить business_metrics.py")
    print("4. Запустить generate_visualizations.py")

if __name__ == "__main__":
    main()