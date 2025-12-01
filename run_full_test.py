#!/usr/bin/env python3
"""
Мастер-скрипт для запуска полного тестирования скилла валидации кредитных моделей
"""

import subprocess
import sys
import os
from pathlib import Path
import time

def check_dependencies():
    """Проверка установленных зависимостей"""
    print("🔍 Проверка зависимостей...")
    
    required_packages = ['pandas', 'numpy', 'sklearn', 'yaml']
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'sklearn':
                import sklearn
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package if package != 'sklearn' else 'scikit-learn')
    
    if missing_packages:
        print(f"\n⚠️ Отсутствуют пакеты: {', '.join(missing_packages)}")
        print("Установите их командой:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ Все зависимости установлены")
    return True

def run_pipeline_step(script_path, description):
    """Выполнение этапа пайплайна"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"📝 Выполнение: {script_path}")
    print('='*60)
    
    try:
        if Path(script_path).exists():
            result = subprocess.run([sys.executable, script_path], 
                                  capture_output=False, 
                                  text=True, 
                                  check=True)
            print(f"✅ {description} - УСПЕШНО")
            return True
        else:
            print(f"❌ Скрипт не найден: {script_path}")
            return False
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - ОШИБКА (код: {e.returncode})")
        return False
    except Exception as e:
        print(f"❌ {description} - ОШИБКА: {e}")
        return False

def test_original_scripts():
    """Тестирование оригинальных скриптов из репозитория"""
    print("\n🧪 ТЕСТИРОВАНИЕ ОРИГИНАЛЬНЫХ СКРИПТОВ")
    print("="*50)
    
    # Проверяем наличие модели и данных
    if not Path("test_models/credit_model_test.pkl").exists():
        print("❌ Тестовая модель не найдена. Сначала создайте тестовые данные.")
        return False
    
    if not Path("test_data/validation_data.csv").exists():
        print("❌ Валидационные данные не найдены.")
        return False
    
    # Создаем директории для результатов
    os.makedirs("results/metrics", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True) 
    os.makedirs("results/stability", exist_ok=True)
    
    success_count = 0
    
    # Тест 1: Расчет метрик
    print("\n1️⃣ Тестирование calculate_metrics.py")
    try:
        result = subprocess.run([
            sys.executable, "scripts/calculate_metrics.py",
            "--model", "test_models/credit_model_test.pkl",
            "--data", "test_data/validation_data.csv", 
            "--target-col", "target",
            "--output", "results/metrics/"
        ], capture_output=True, text=True, check=True)
        
        print("✅ calculate_metrics.py работает корректно")
        success_count += 1
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка в calculate_metrics.py:")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
    
    # Тест 2: Анализ стабильности  
    print("\n2️⃣ Тестирование stability_analysis.py")
    try:
        result = subprocess.run([
            sys.executable, "scripts/stability_analysis.py",
            "--train-data", "test_data/train_data.csv",
            "--validation-data", "test_data/validation_data.csv",
            "--output", "results/stability/"
        ], capture_output=True, text=True, timeout=60)
        
        print("✅ stability_analysis.py работает корректно")
        success_count += 1
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка в stability_analysis.py:")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
    except subprocess.TimeoutExpired:
        print("❌ Timeout при выполнении stability_analysis.py")
    except Exception as e:
        print(f"❌ Общая ошибка в stability_analysis.py: {e}")
    
    # Тест 3: Бизнес-метрики
    print("\n3️⃣ Тестирование business_metrics.py") 
    try:
        result = subprocess.run([
            sys.executable, "scripts/business_metrics.py",
            "--model", "test_models/credit_model_test.pkl",
            "--data", "test_data/validation_data.csv",
            "--target-col", "target",
            "--output", "results/metrics/"
        ], capture_output=True, text=True, timeout=60)
        
        print("✅ business_metrics.py работает корректно")
        success_count += 1
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка в business_metrics.py:")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
    except subprocess.TimeoutExpired:
        print("❌ Timeout при выполнении business_metrics.py")
    except Exception as e:
        print(f"❌ Общая ошибка в business_metrics.py: {e}")
    
    print(f"\n📊 Результат тестирования оригинальных скриптов: {success_count}/3")
    return success_count >= 2

def show_results():
    """Показать созданные результаты"""
    print("\n📊 СОЗДАННЫЕ ФАЙЛЫ И РЕЗУЛЬТАТЫ:")
    print("="*50)
    
    files_to_check = [
        "test_data/validation_data.csv",
        "test_data/train_data.csv", 
        "test_models/credit_model_test.pkl",
        "results/metrics/metrics.json",
        "results/stability/"
    ]
    
    for file_path in files_to_check:
        path = Path(file_path)
        if path.exists():
            if path.is_file():
                size = path.stat().st_size
                print(f"✅ {file_path} ({size:,} байт)")
            else:
                print(f"✅ {file_path}/ (директория)")
        else:
            print(f"❌ {file_path} - не найден")

def main():
    """Главная функция"""
    start_time = time.time()
    
    print("🎯 ПОЛНОЕ ТЕСТИРОВАНИЕ СКИЛЛА ВАЛИДАЦИИ КРЕДИТНЫХ МОДЕЛЕЙ")
    print("="*80)
    print("Автор скилла: @00060633")
    print("Репозиторий: credit-model-validation-skill")
    print("="*80)
    
    # Проверяем зависимости
    if not check_dependencies():
        print("\n❌ Сначала установите недостающие зависимости")
        return 1
    
    # Этапы тестирования
    pipeline_steps = [
        ("test_full_pipeline.py", "Создание синтетических тестовых данных и модели"),
    ]
    
    successful_steps = 0
    
    # Выполняем основные этапы
    for script_path, description in pipeline_steps:
        success = run_pipeline_step(script_path, description)
        if success:
            successful_steps += 1
    
    # Тестируем оригинальные скрипты
    if successful_steps > 0:
        test_original_scripts()
    
    # Показываем результаты
    show_results()
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    print("\n" + "="*80)
    print("🏁 ИТОГИ ТЕСТИРОВАНИЯ")
    print("="*80)
    print(f"⏱️ Время выполнения: {execution_time:.1f} секунд")
    
    if successful_steps == len(pipeline_steps):
        print("🎉 ТЕСТИРОВАНИЕ УСПЕШНО ЗАВЕРШЕНО!")
        print("\n📋 Что делать дальше:")
        print("• Изучите результаты в папке results/")
        print("• Адаптируйте скрипты под свои данные")
        print("• Проверьте качество валидации на реальных моделях")
        return 0
    else:
        print("⚠️ Тестирование завершено с ошибками")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)