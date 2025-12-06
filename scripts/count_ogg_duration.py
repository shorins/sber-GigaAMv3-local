import os
import argparse
import torchaudio
from pathlib import Path
from tqdm import tqdm

def get_total_duration(folder_path):
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Ошибка: Папка '{folder_path}' не существует.")
        return

    # Поиск всех .ogg файлов (без учета регистра)
    ogg_files = sorted(list(folder.glob("*.ogg")))
    
    if not ogg_files:
        print(f"В папке '{folder_path}' нет файлов .ogg.")
        return

    total_seconds = 0.0
    print(f"Найдено {len(ogg_files)} файлов .ogg. Подсчет длительности...")

    # Используем tqdm для отображения прогресса
    for file_path in tqdm(ogg_files, unit="file"):
        try:
            # Получаем метаданные файла без полной загрузки
            metadata = torchaudio.info(str(file_path))
            
            if metadata.sample_rate > 0:
                duration = metadata.num_frames / metadata.sample_rate
                total_seconds += duration
            else:
                print(f"Предупреждение: Некорректный sample_rate для {file_path.name}")
                
        except Exception as e:
            print(f"Ошибка при чтении {file_path.name}: {e}")

    # Форматирование времени
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    milliseconds = int((total_seconds - int(total_seconds)) * 1000)

    print("-" * 30)
    print(f"Общая длительность:")
    print(f"Секунды: {total_seconds:.3f}")
    print(f"Формат: {hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}")
    print("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Подсчет общей длительности .ogg файлов в папке.")
    parser.add_argument("folder_path", type=str, help="Путь к папке с .ogg файлами")
    
    args = parser.parse_args()
    get_total_duration(args.folder_path)
