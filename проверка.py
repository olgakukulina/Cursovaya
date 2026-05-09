import json
import csv
import numpy as np
import matplotlib.pyplot as plt

class Appliance:
        expected_params = {
            "washing_machine": {"Максимальная загрузка, кг", "Максимальная скорость отжима, об/мин", "Уровень шума, дБ", "Цена, руб"},
            "vacuum_cleaner": {"Мощность всасывания, Вт", "Емкость пылесборника, л", "Уровень шума, дБ", "Радиус действия, м", "Цена, руб"},
            "multicooker": {"Объем чаши, л", "Мощность, Вт", "Количество автоматических программ, шт", "Вес, кг", "Цена, руб"}
        }

        def __init__(self, name: str, appliance_type: str, characteristics: dict):
            self._validate(appliance_type, characteristics)
            self.name = name.strip()
            self.type = appliance_type
            self.characteristics = characteristics
            self.tech_level = None

        def _validate(self, appliance_type: str, data: dict):
            if appliance_type not in self.expected_params:
                raise ValueError(f"Неизвестный тип: '{appliance_type}'")
            required = self.expected_params[appliance_type]
            missing = required - data.keys()
            if missing:
                raise ValueError(f"Нет обязательных параметров: {missing}")
            for k, v in data.items():
                if not isinstance(v, (int, float)):
                    raise ValueError(f"'{k}' должно быть числом, получено {type(v).__name__}")
                if v <= 0:
                    raise ValueError(f"'{k}' не может быть <= 0")



def _parse_json(filepath, appliance_type):
    appliances = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("JSON должен содержать массив объектов.")
            for appliance in data:
                if not isinstance(appliance, dict):
                    raise ValueError(f"Не является объектом (словарём).")
                if "Название" not in appliance:
                    raise ValueError(f"Отсутствует ключ 'Название'.")
                name = appliance.pop("Название")
                appliance = {k: float(v) for k, v in appliance.items()}
                appliances.append(Appliance(name, appliance_type, appliance))
    except FileNotFoundError:
        print("Файл не найден.")
    except Exception as e:
        print(f"Ошибка: {e}")
    return appliances

def _parse_csv(filepath: str, appliance_type: str) -> list:
    appliances = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=";")
            if reader.fieldnames is None or "Название" not in reader.fieldnames:
                raise ValueError("CSV должен содержать колонку 'Название'.")
            for row_num, row in enumerate(reader, start=2):
                name = row.pop("Название", "").strip()
                if not name:
                    raise ValueError(f"Пустое название в строке {row_num}.")
                clean_params = {k: float(v) for k, v in row.items()}
                appliances.append(Appliance(name, appliance_type, clean_params))
    except FileNotFoundError:
        print("Файл не найден.")
    except Exception as e:
        print(f"Ошибка: {e}")
    return appliances


def _parse_txt(filepath: str, appliance_type: str) -> list:
    appliances = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if not lines or not lines[0].strip():
            raise ValueError("Файл пуст или не содержит заголовков.")
        headers = [h.strip() for h in lines[0].split(";")]
        if "Название" not in headers:
            raise ValueError("В первой строке должна быть колонка 'Название'.")
        for line_num, line in enumerate(lines[1:], start=2):
            line = line.strip()
            if not line:
                continue
            values = [v.strip() for v in line.split(";")]
            if len(values) != len(headers):
                raise ValueError(f"Несоответствие количества полей в строке {line_num}.")
            row = dict(zip(headers, values))
            name = row.pop("Название", "").strip()
            if not name:
                raise ValueError(f"Пустое название в строке {line_num}.")
            clean_params = {k: float(v) for k, v in row.items()}
            appliances.append(Appliance(name, appliance_type, clean_params))
    except FileNotFoundError:
        print("Файл не найден.")
    except Exception as e:
        print(f"Ошибка: {e}")
    return appliances


def normalize_for_radar(samples: list, appliance_type: str):
    if not samples:
        return {}
    worse_is_better = {
        "washing_machine": {"Уровень шума, дБ", "Цена, руб"},
        "vacuum_cleaner": {"Уровень шума, дБ", "Цена, руб"},
        "multicooker": {"Уровень шума, дБ", "Цена, руб", "Вес, кг"}
    }
    bad_params = worse_is_better.get(appliance_type, set())
    param_values = {}
    for sample in samples:
        for param, value in sample.characteristics.items():
            if param not in param_values:
                param_values[param] = []
            param_values[param].append(value)
    extremes = {}
    for param, values in param_values.items():
        extremes[param] = {"min": min(values), "max": max(values)}
    normalized = {}
    for sample in samples:
        normalized[sample.name] = {}
        for param, value in sample.characteristics.items():
            if param in bad_params:
                if extremes[param]["min"] == 0:
                    normalized[sample.name][param] = 1.0 if value == 0 else 0.0
                else:
                    normalized[sample.name][param] = round(extremes[param]["min"] / value, 2)
            else:
                if extremes[param]["max"] == 0:
                    normalized[sample.name][param] = 0.0
                else:
                    normalized[sample.name][param] = round(value / extremes[param]["max"], 2)
    return normalized

def calculate_tech_level_simple(normalized: dict) -> dict:
    result = {}
    for name, params in normalized.items():
        values = list(params.values())
        if not values:
            result[name] = 0.0
        else:
            result[name] = round(sum(values) / len(values), 2)
    return result


def plot_tech_levels_bar(tech_levels: dict, title: str = "Технический уровень образцов"):
    sorted_items = sorted(tech_levels.items(), key=lambda x: x[1], reverse=True)
    names = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, values, color='#4C72B0', edgecolor='black')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.02,
                 f'{height:.2f}', ha='center', va='bottom', fontsize=10)
    plt.ylabel("Технический уровень", fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylim(0, 1.1)  # Запас сверху для подписей
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()


def create_radial(models: list, params: list, values: list):
    for v in values:
        v += v[:1]
    angles = np.linspace(0, 2 * np.pi, len(params), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for i in range(len(models)):
        ax.plot(angles, values[i], 'o-', linewidth=2, label=models[i], color=colors[i % len(colors)])
        ax.fill(angles, values[i], alpha=0.15, color=colors[i % len(colors)])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(params, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title("Сравнение относительных характеристик", pad=20)
    plt.show()

normalized = normalize_for_radar(_parse_json("prov.json", "washing_machine"), "washing_machine")
print(normalize_for_radar(_parse_json("prov.json", "washing_machine"), "washing_machine"))

models = list(normalized.keys())
params = list(normalized[models[0]].keys())
values = [list(sample.values()) for sample in normalized.values()]
create_radial(models, params, values)


plot_tech_levels_bar(calculate_tech_level_simple(normalize_for_radar(_parse_json("prov.json", "washing_machine"), "washing_machine")))