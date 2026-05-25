import customtkinter as ctk
from tkinter import messagebox, filedialog
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy.optimize import linprog
import time

ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")

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

def normalize_relative_to_first(samples: list, appliance_type: str):
    if len(samples) < 1:
        return {}
    worse_is_better = {
        "washing_machine": {"Уровень шума, дБ", "Цена, руб"},
        "vacuum_cleaner": {"Уровень шума, дБ", "Цена, руб"},
        "multicooker": {"Уровень шума, дБ", "Цена, руб", "Вес, кг"}
    }
    bad_params = worse_is_better.get(appliance_type, set())
    base = samples[0]
    normalized = {}
    for sample in samples:
        normalized[sample.name] = {}
        for param, value in sample.characteristics.items():
            base_val = base.characteristics[param]
            if base_val == 0:
                q = 1.0
            else:
                if param in bad_params:
                    q = base_val / value
                else:
                    q = value / base_val
            normalized[sample.name][param] = round(q, 4)
    return normalized

def calculate_weighted_tech_level(normalized: dict, weights: dict):
    result = {}
    for name, params in normalized.items():
        total = 0.0
        total_weight = 0.0
        for param, q in params.items():
            w = weights.get(param, 0)
            total += w * q
            total_weight += w
        if total_weight == 0:
            result[name] = 0.0
        else:
            result[name] = round(total / total_weight, 4)
    return result

def plot_tech_levels_bar(tech_levels: dict, title: str = "Технический уровень образцов"):
    sorted_items = sorted(tech_levels.items(), key=lambda x: x[1], reverse=True)
    names = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, values, color='#FF69B4', edgecolor='black')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.02,
                 f'{height:.2f}', ha='center', va='bottom', fontsize=10)
    plt.ylabel("Технический уровень", fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylim(0, max(1.1, max(values)*1.1))
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()

def create_radial(models: list, params: list, values: list):
    for v in values:
        v.append(v[0])
    angles = np.linspace(0, 2 * np.pi, len(params), endpoint=False).tolist()
    angles.append(angles[0])
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))
    colors = ['#FF69B4', '#FF1493', '#C71585', '#DB7093', '#FFB6C1']
    for i in range(len(models)):
        ax.plot(angles, values[i], 'o-', linewidth=2, label=models[i], color=colors[i % len(colors)])
        ax.fill(angles, values[i], alpha=0.15, color=colors[i % len(colors)])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(params, fontsize=10)
    ax.set_ylim(0, max(max(v) for v in values) * 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title("Сравнение относительных характеристик (база = первый образец)", pad=20)
    plt.show()

def plot_bubble_chart(appliances, tech_levels, appliance_type):
    if not appliances or not tech_levels:
        print("Нет данных для пузырьковой диаграммы")
        return

    size_param_map = {
        "washing_machine": "Максимальная загрузка, кг",
        "vacuum_cleaner": "Мощность всасывания, Вт",
        "multicooker": "Объем чаши, л"
    }
    size_key = size_param_map.get(appliance_type, list(appliances[0].characteristics.keys())[0])

    names = []
    tech_vals = []
    prices = []
    sizes = []

    for app in appliances:
        names.append(app.name)
        tech_vals.append(tech_levels.get(app.name, 0))
        price = app.characteristics.get("Цена, руб", 0)
        prices.append(price)
        size_val = app.characteristics.get(size_key, 1)
        sizes.append(size_val * 10)

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(tech_vals, prices, s=sizes, c=tech_vals, cmap='RdPu', alpha=0.7, edgecolors='black')
    for i, name in enumerate(names):
        plt.annotate(name, (tech_vals[i], prices[i]), xytext=(5, 5),
                     textcoords='offset points', fontsize=8, alpha=0.8)
    plt.xlabel("Технический уровень (относительно эталона)", fontsize=12)
    plt.ylabel("Цена, руб", fontsize=12)
    plt.title(f"Пузырьковая диаграмма: ТУ vs Цена ({appliance_type})\nРазмер пузырька = {size_key}", fontsize=14)
    plt.grid(True, alpha=0.3, linestyle='--')
    cbar = plt.colorbar(scatter, label="Технический уровень")
    cbar.ax.set_ylabel("ТУ", rotation=270, labelpad=15)
    plt.tight_layout()
    plt.show()

TYPE_MAPPING = {
    "Стиральные машины": "washing_machine",
    "Пылесосы": "vacuum_cleaner",
    "Мультиварки": "multicooker"
}

PARAMS_MAP = {
    "washing_machine": ["Максимальная загрузка, кг", "Максимальная скорость отжима, об/мин", "Уровень шума, дБ",
                        "Цена, руб"],
    "vacuum_cleaner": ["Мощность всасывания, Вт", "Емкость пылесборника, л", "Уровень шума, дБ", "Радиус действия, м",
                       "Цена, руб"],
    "multicooker": ["Объем чаши, л", "Мощность, Вт", "Количество автоматических программ, шт", "Вес, кг", "Цена, руб"]
}

WEIGHTS = {
    "washing_machine": {
        "Максимальная загрузка, кг": 0.25,
        "Максимальная скорость отжима, об/мин": 0.20,
        "Уровень шума, дБ": 0.20,
        "Цена, руб": 0.35
    },
    "vacuum_cleaner": {
        "Мощность всасывания, Вт": 0.30,
        "Емкость пылесборника, л": 0.10,
        "Уровень шума, дБ": 0.15,
        "Радиус действия, м": 0.10,
        "Цена, руб": 0.35
    },
    "multicooker": {
        "Объем чаши, л": 0.20,
        "Мощность, Вт": 0.10,
        "Количество автоматических программ, шт": 0.25,
        "Вес, кг": 0.15,
        "Цена, руб": 0.30
    }
}

class TechLevelApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Оценка технического уровня бытовой техники")
        self.geometry("880x720")
        self.configure(fg_color="#FFE4E1")
        self.appliances = []
        self.entries = {}
        self.type_var = ctk.StringVar(value="Стиральные машины")
        self.current_mode = "calc"
        self.forecast_model = None
        self.forecast_data = None
        self.forecast_features = ['steel_price', 'plastic_price', 'is_pre_holiday_season', 'production_capacity_utilization', 'avg_worker_skill_level']
        self.forecast_target = 'monthly_appliance_production'

        self.setup_ui()

    def setup_ui(self):
        self.main_frame = ctk.CTkFrame(self, fg_color="#FFE4E1")
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=10)

        mode_frame = ctk.CTkFrame(self.main_frame, fg_color="#FFF0F5")
        mode_frame.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(mode_frame, text="Режим работы:", text_color="#8B008B", font=ctk.CTkFont(size=14)).pack(side="left", padx=10)
        self.mode_selector = ctk.CTkSegmentedButton(
            mode_frame,
            values=["Расчёт ТУ", "Прогнозирование", "Оптимизация"],
            command=self.change_mode,
            fg_color="#FF69B4",
            selected_color="#FF1493",
            text_color="white",
            selected_hover_color="#C71585"
        )
        self.mode_selector.set("Расчёт ТУ")
        self.mode_selector.pack(side="left", padx=10, expand=True, fill="x")

        self.content_frame = ctk.CTkFrame(self.main_frame, fg_color="#FFE4E1")
        self.content_frame.pack(fill="both", expand=True)

        self.calc_frame = ctk.CTkFrame(self.content_frame, fg_color="#FFE4E1")
        self.setup_calc_ui()
        self.forecast_frame = ctk.CTkFrame(self.content_frame, fg_color="#FFF0F5")
        self.setup_forecast_ui()
        self.optim_frame = ctk.CTkFrame(self.content_frame, fg_color="#FFF0F5")
        self.setup_optim_ui()

        self.change_mode("Расчёт ТУ")

    def setup_calc_ui(self):
        top_frame = ctk.CTkFrame(self.calc_frame, fg_color="#FFF0F5")
        top_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(top_frame, text="Тип техники:", text_color="#8B008B").pack(side="left", padx=5)
        self.type_menu = ctk.CTkOptionMenu(top_frame, values=list(TYPE_MAPPING.keys()), variable=self.type_var,
                                           command=self.on_type_change, fg_color="#FF69B4", button_color="#FF1493")
        self.type_menu.pack(side="left", padx=5)

        ctk.CTkButton(top_frame, text="Загрузить из файла", command=self.load_from_file,
                      fg_color="#FF69B4", hover_color="#FF1493", text_color="white").pack(side="right", padx=5)
        ctk.CTkButton(top_frame, text="Добавить вручную", command=self.add_manually,
                      fg_color="#FF69B4", hover_color="#FF1493", text_color="white").pack(side="right", padx=5)

        self.input_frame = ctk.CTkFrame(self.calc_frame, fg_color="#FFF0F5")
        self.input_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        self.build_input_form("Стиральные машины")

        self.list_frame = ctk.CTkFrame(self.calc_frame, fg_color="#FFF0F5")
        self.list_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        ctk.CTkLabel(self.list_frame, text="Загруженные образцы:", font=ctk.CTkFont(weight="bold"), text_color="#8B008B").pack(pady=5)
        self.list_text = ctk.CTkTextbox(self.list_frame, width=350, fg_color="#FFFFFF", text_color="#000000")
        self.list_text.pack(padx=10, pady=5, fill="both", expand=True)
        ctk.CTkButton(self.list_frame, text="Очистить список", command=self.clear_list,
                      fg_color="#FF69B4", hover_color="#FF1493", text_color="white").pack(pady=5)

        calc_buttons = ctk.CTkFrame(self.calc_frame, fg_color="#FFE4E1")
        calc_buttons.pack(side="bottom", fill="x", padx=10, pady=10)

        ctk.CTkButton(calc_buttons, text="Рассчитать ТУ и построить графики",
                      fg_color="#2b8a3e", hover_color="#1c6f30", text_color="white",
                      command=self.run_calculation).pack(fill="x", pady=2)

        ctk.CTkButton(calc_buttons, text="Пузырьковая диаграмма",
                      fg_color="#FF69B4", hover_color="#FF1493", text_color="white",
                      command=self.run_bubble_chart).pack(fill="x", pady=2)

    def setup_forecast_ui(self):
        main_container = ctk.CTkFrame(self.forecast_frame, fg_color="#FFF0F5")
        main_container.pack(fill="both", expand=True, padx=20, pady=20)

        ctk.CTkLabel(main_container, text="Прогнозирование объёма производства", font=ctk.CTkFont(size=18, weight="bold"), text_color="#8B008B").pack(pady=10)

        self.forecast_status = ctk.CTkLabel(main_container, text="", text_color="#2b8a3e")
        self.forecast_status.pack(pady=5)

        load_btn = ctk.CTkButton(main_container, text="Загрузить данные для обучения", command=self.load_production_data,
                                 fg_color="#FF69B4", hover_color="#FF1493", text_color="white")
        load_btn.pack(pady=5)

        self.feature_entries = {}
        feature_labels = ["Цена стали (руб/кг)", "Цена пластика (руб/кг)", "Предпраздничный сезон (0/1)", "Загрузка мощностей (0-100)", "Уровень квалификации (1-10)"]
        internal_names = self.forecast_features
        for label, name in zip(feature_labels, internal_names):
            row = ctk.CTkFrame(main_container, fg_color="#FFF0F5")
            row.pack(fill="x", padx=10, pady=5)
            ctk.CTkLabel(row, text=label, width=250, anchor="w", text_color="#8B008B").pack(side="left", padx=5)
            entry = ctk.CTkEntry(row, fg_color="#FFFFFF", text_color="#000000")
            entry.pack(side="right", padx=5, fill="x", expand=True)
            self.feature_entries[name] = entry

        predict_btn = ctk.CTkButton(main_container, text="Получить прогноз по введённым значениям", command=self.predict_production,
                                    fg_color="#2b8a3e", hover_color="#1c6f30", text_color="white")
        predict_btn.pack(pady=10)

        self.prediction_label = ctk.CTkLabel(main_container, text="", font=ctk.CTkFont(size=14), text_color="#8B008B")
        self.prediction_label.pack(pady=5)

        interval_frame = ctk.CTkFrame(main_container, fg_color="#FFF0F5", border_width=1, border_color="#FF69B4")
        interval_frame.pack(fill="x", padx=10, pady=15)

        ctk.CTkLabel(interval_frame, text="Интервальный анализ", font=ctk.CTkFont(weight="bold"), text_color="#8B008B").pack(pady=5)

        select_row = ctk.CTkFrame(interval_frame, fg_color="#FFF0F5")
        select_row.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(select_row, text="Признак для анализа:", text_color="#8B008B").pack(side="left", padx=5)
        interval_features = [f for f in self.forecast_features if f != 'is_pre_holiday_season']
        self.interval_feature = ctk.CTkOptionMenu(select_row, values=interval_features,
                                                  fg_color="#FF69B4", button_color="#FF1493", text_color="white")
        self.interval_feature.pack(side="right", padx=5, fill="x", expand=True)

        range_row = ctk.CTkFrame(interval_frame, fg_color="#FFF0F5")
        range_row.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(range_row, text="Мин:", text_color="#8B008B").pack(side="left", padx=5)
        self.min_entry = ctk.CTkEntry(range_row, width=100, fg_color="#FFFFFF", text_color="#000000")
        self.min_entry.pack(side="left", padx=5)
        ctk.CTkLabel(range_row, text="Макс:", text_color="#8B008B").pack(side="left", padx=5)
        self.max_entry = ctk.CTkEntry(range_row, width=100, fg_color="#FFFFFF", text_color="#000000")
        self.max_entry.pack(side="left", padx=5)
        self.plot_btn = ctk.CTkButton(range_row, text="Построить график зависимости", command=self.plot_dependence,
                                      fg_color="#FF69B4", hover_color="#FF1493", text_color="white")
        self.plot_btn.pack(side="right", padx=10)

        ctk.CTkLabel(interval_frame, text="Остальные признаки фиксируются по значениям из полей выше", font=ctk.CTkFont(size=10), text_color="#8B008B").pack(pady=2)

    def setup_optim_ui(self):
        main_container = ctk.CTkFrame(self.optim_frame, fg_color="#FFF0F5")
        main_container.pack(fill="both", expand=True, padx=20, pady=20)

        ctk.CTkLabel(main_container, text="Транспортная оптимизация", font=ctk.CTkFont(size=18, weight="bold"), text_color="#8B008B").pack(pady=10)

        param_frame = ctk.CTkFrame(main_container, fg_color="#FFE4E1")
        param_frame.pack(fill="x", padx=10, pady=10)

        row1 = ctk.CTkFrame(param_frame, fg_color="#FFF0F5")
        row1.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(row1, text="Количество заводов:", text_color="#8B008B").pack(side="left", padx=5)
        self.num_factories = ctk.CTkEntry(row1, width=100, fg_color="#FFFFFF", text_color="#000000")
        self.num_factories.pack(side="left", padx=5)
        ctk.CTkLabel(row1, text="Количество торговых сетей:", text_color="#8B008B").pack(side="left", padx=5)
        self.num_networks = ctk.CTkEntry(row1, width=100, fg_color="#FFFFFF", text_color="#000000")
        self.num_networks.pack(side="left", padx=5)
        ctk.CTkButton(row1, text="Задать размерность", command=self.init_optim_matrices,
                      fg_color="#FF69B4", hover_color="#FF1493", text_color="white").pack(side="left", padx=10)

        self.optim_params_frame = ctk.CTkFrame(main_container, fg_color="#FFE4E1")
        self.optim_params_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.optim_entries = {}
        self.supply_entries = []
        self.demand_entries = []
        self.cost_entries = []

        self.solve_btn = ctk.CTkButton(main_container, text="Решить транспортную задачу", command=self.solve_transport_problem,
                                       fg_color="#2b8a3e", hover_color="#1c6f30", text_color="white", state="disabled")
        self.solve_btn.pack(pady=10)

        self.optim_result_text = ctk.CTkTextbox(main_container, height=250, fg_color="#FFFFFF", text_color="#000000")
        self.optim_result_text.pack(fill="both", expand=True, padx=10, pady=10)

    def init_optim_matrices(self):
        try:
            n_factories = int(self.num_factories.get())
            n_networks = int(self.num_networks.get())
            if n_factories <= 0 or n_networks <= 0:
                raise ValueError("Количество должно быть положительным")
            if n_factories > 10 or n_networks > 10:
                if not messagebox.askyesno("Предупреждение", f"Большая размерность ({n_factories}x{n_networks}) может быть неудобной. Продолжить?"):
                    return
        except ValueError as e:
            messagebox.showerror("Ошибка", f"Некорректный ввод размерности: {e}")
            return

        for widget in self.optim_params_frame.winfo_children():
            widget.destroy()

        self.optim_entries = {
            "supply": [],
            "demand": [],
            "costs": []
        }

        scroll_frame = ctk.CTkScrollableFrame(self.optim_params_frame, fg_color="#FFF0F5", height=300)
        scroll_frame.pack(fill="both", expand=True)

        supply_frame = ctk.CTkFrame(scroll_frame, fg_color="#FFF0F5")
        supply_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(supply_frame, text="Мощности заводов (ед./мес):", text_color="#8B008B", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=5)
        supply_entries_frame = ctk.CTkFrame(supply_frame, fg_color="#FFF0F5")
        supply_entries_frame.pack(fill="x", padx=10, pady=5)
        for i in range(n_factories):
            row = ctk.CTkFrame(supply_entries_frame, fg_color="#FFF0F5")
            row.pack(fill="x", pady=2)
            ctk.CTkLabel(row, text=f"Завод Z{i+1}:", width=100, text_color="#8B008B").pack(side="left", padx=5)
            entry = ctk.CTkEntry(row, fg_color="#FFFFFF", text_color="#000000")
            entry.pack(side="left", padx=5, fill="x", expand=True)
            self.supply_entries.append(entry)

        demand_frame = ctk.CTkFrame(scroll_frame, fg_color="#FFF0F5")
        demand_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(demand_frame, text="Спрос торговых сетей (ед./мес):", text_color="#8B008B", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=5)
        demand_entries_frame = ctk.CTkFrame(demand_frame, fg_color="#FFF0F5")
        demand_entries_frame.pack(fill="x", padx=10, pady=5)
        for j in range(n_networks):
            row = ctk.CTkFrame(demand_entries_frame, fg_color="#FFF0F5")
            row.pack(fill="x", pady=2)
            ctk.CTkLabel(row, text=f"Сеть N{j+1}:", width=100, text_color="#8B008B").pack(side="left", padx=5)
            entry = ctk.CTkEntry(row, fg_color="#FFFFFF", text_color="#000000")
            entry.pack(side="left", padx=5, fill="x", expand=True)
            self.demand_entries.append(entry)

        costs_frame = ctk.CTkFrame(scroll_frame, fg_color="#FFF0F5")
        costs_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(costs_frame, text="Матрица затрат (руб./ед.):", text_color="#8B008B", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=5)
        costs_entries_frame = ctk.CTkFrame(costs_frame, fg_color="#FFF0F5")
        costs_entries_frame.pack(fill="x", padx=10, pady=5)

        header_row = ctk.CTkFrame(costs_entries_frame, fg_color="#FFF0F5")
        header_row.pack(fill="x", pady=2)
        ctk.CTkLabel(header_row, text="Завод\\Сеть", width=100, text_color="#8B008B", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=5)
        for j in range(n_networks):
            ctk.CTkLabel(header_row, text=f"N{j+1}", width=80, text_color="#8B008B", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=5)

        self.cost_entries = []
        for i in range(n_factories):
            row = ctk.CTkFrame(costs_entries_frame, fg_color="#FFF0F5")
            row.pack(fill="x", pady=2)
            ctk.CTkLabel(row, text=f"Z{i+1}", width=100, text_color="#8B008B").pack(side="left", padx=5)
            row_entries = []
            for j in range(n_networks):
                entry = ctk.CTkEntry(row, width=80, fg_color="#FFFFFF", text_color="#000000")
                entry.pack(side="left", padx=5)
                row_entries.append(entry)
            self.cost_entries.append(row_entries)

        self.solve_btn.configure(state="normal")

    def solve_transport_problem(self):
        try:
            n_factories = len(self.supply_entries)
            n_networks = len(self.demand_entries)

            supply = []
            for i, entry in enumerate(self.supply_entries):
                val = float(entry.get())
                if val <= 0:
                    raise ValueError(f"Мощность завода Z{i+1} должна быть положительной")
                supply.append(val)

            demand = []
            for j, entry in enumerate(self.demand_entries):
                val = float(entry.get())
                if val <= 0:
                    raise ValueError(f"Спрос сети N{j+1} должен быть положительным")
                demand.append(val)

            total_supply = sum(supply)
            total_demand = sum(demand)

            costs = []
            for i in range(n_factories):
                row = []
                for j in range(n_networks):
                    val = float(self.cost_entries[i][j].get())
                    if val <= 0:
                        raise ValueError(f"Затраты Z{i+1}->N{j+1} должны быть положительными")
                    row.append(val)
                costs.append(row)
            costs = np.array(costs, dtype=np.float64)

            self.optim_result_text.delete("1.0", "end")
            self.optim_result_text.insert("end", "=" * 70 + "\n")
            self.optim_result_text.insert("end", "ТРАНСПОРТНАЯ ЗАДАЧА\n")
            self.optim_result_text.insert("end", "=" * 70 + "\n\n")
            self.optim_result_text.insert("end", f"Суммарная мощность: {total_supply:.0f} ед.\n")
            self.optim_result_text.insert("end", f"Суммарный спрос: {total_demand:.0f} ед.\n")

            if total_supply < total_demand:
                self.optim_result_text.insert("end", f"ПРЕДУПРЕЖДЕНИЕ: Спрос превышает мощность на {total_demand - total_supply:.0f} ед.\n")
            elif total_supply > total_demand:
                self.optim_result_text.insert("end", f"ПРЕДУПРЕЖДЕНИЕ: Мощность превышает спрос на {total_supply - total_demand:.0f} ед.\n")
            self.optim_result_text.insert("end", "\n")

            c = costs.flatten()
            n_vars = n_factories * n_networks

            A_ub = []
            b_ub = []
            for i in range(n_factories):
                row = [0] * n_vars
                for j in range(n_networks):
                    row[i * n_networks + j] = 1
                A_ub.append(row)
                b_ub.append(supply[i])

            A_eq = []
            b_eq = []
            for j in range(n_networks):
                row = [0] * n_vars
                for i in range(n_factories):
                    row[i * n_networks + j] = 1
                A_eq.append(row)
                b_eq.append(demand[j])

            bounds = [(0, None)] * n_vars

            methods = ['highs', 'highs-ipm', 'highs-ds']
            best_result = None
            best_time = float('inf')

            for m in methods:
                t_start = time.perf_counter()
                r = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=m)
                t_end = time.perf_counter()
                if r.success and (t_end - t_start) < best_time:
                    best_time = t_end - t_start
                    best_result = r

            if best_result is None or not best_result.success:
                self.optim_result_text.insert("end", "Оптимальное решение не найдено.\n")
                if best_result is not None:
                    self.optim_result_text.insert("end", f"Сообщение: {best_result.message}\n")
                return

            result = best_result
            x_opt = result.x
            total_cost = result.fun

            self.optim_result_text.insert("end", f"ОПТИМАЛЬНЫЕ ЗАТРАТЫ: {total_cost:,.0f} руб.\n\n")

            solution_matrix = x_opt.reshape(n_factories, n_networks)

            self.optim_result_text.insert("end", "РАСПРЕДЕЛЕНИЕ ПОСТАВОК:\n")
            self.optim_result_text.insert("end", "-" * 70 + "\n")
            for i in range(n_factories):
                row_sum = 0
                for j in range(n_networks):
                    val = solution_matrix[i, j]
                    if val > 1e-6:
                        cost_part = val * costs[i, j]
                        self.optim_result_text.insert("end", f"  Z{i+1} -> N{j+1}: {val:6.0f} ед. (затраты {cost_part:12,.0f} руб.)\n")
                        row_sum += val
                self.optim_result_text.insert("end", f"  Итого с завода Z{i+1}: {row_sum:.0f} ед.\n\n")

            self.optim_result_text.insert("end", "ЗАГРУЗКА ЗАВОДОВ:\n")
            for i in range(n_factories):
                used = solution_matrix[i].sum()
                cap = supply[i]
                util = (used / cap) * 100 if cap > 0 else 0
                self.optim_result_text.insert("end", f"  Z{i+1}: {used:.0f}/{cap:.0f} ед. ({util:.1f}%)\n")

            self.optim_result_text.insert("end", "\nУДОВЛЕТВОРЕНИЕ СПРОСА:\n")
            for j in range(n_networks):
                received = solution_matrix[:, j].sum()
                req = demand[j]
                self.optim_result_text.insert("end", f"  N{j+1}: получено {received:.0f}, требуется {req:.0f} ед.\n")

            calc_cost = np.sum(solution_matrix * costs)
            self.optim_result_text.insert("end", f"\nРучной пересчёт затрат: {calc_cost:,.0f} руб.\n")
            self.optim_result_text.insert("end", "=" * 70 + "\n")
            self.optim_result_text.insert("end", "Решение найдено. Распределение оптимально.\n")

        except ValueError as e:
            messagebox.showerror("Ошибка ввода", str(e))
            self.optim_result_text.insert("end", f"Ошибка: {str(e)}\n")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
            self.optim_result_text.insert("end", f"Ошибка: {str(e)}\n")

    def change_mode(self, mode):
        self.calc_frame.pack_forget()
        self.forecast_frame.pack_forget()
        self.optim_frame.pack_forget()
        if mode == "Расчёт ТУ":
            self.calc_frame.pack(fill="both", expand=True)
            self.current_mode = "calc"
        elif mode == "Прогнозирование":
            self.forecast_frame.pack(fill="both", expand=True)
            self.current_mode = "forecast"
            if self.forecast_model is None:
                self.forecast_status.configure(text="Модель не обучена. Загрузите данные.")
        elif mode == "Оптимизация":
            self.optim_frame.pack(fill="both", expand=True)
            self.current_mode = "optim"

    def on_type_change(self, choice):
        if self.current_mode != "calc":
            return
        self.build_input_form(choice)

    def build_input_form(self, type_name: str):
        for widget in self.input_frame.winfo_children():
            widget.destroy()
        self.entries.clear()

        ctk.CTkLabel(self.input_frame, text=f"Ввод параметров: {type_name}", font=ctk.CTkFont(weight="bold"), text_color="#8B008B").pack(pady=5)

        internal_type = TYPE_MAPPING[type_name]
        params = PARAMS_MAP[internal_type]

        for i, param in enumerate(params):
            row = ctk.CTkFrame(self.input_frame, fg_color="#FFF0F5")
            row.pack(fill="x", padx=10, pady=2)

            ctk.CTkLabel(row, text=f"{param}:", width=200, anchor="w", text_color="#8B008B").pack(side="left", padx=5)
            entry = ctk.CTkEntry(row, placeholder_text="Введите число", fg_color="#FFFFFF", text_color="#000000")
            entry.pack(side="right", padx=5, fill="x", expand=True)
            self.entries[param] = entry

        name_row = ctk.CTkFrame(self.input_frame, fg_color="#FFF0F5")
        name_row.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(name_row, text="Название образца:", width=200, anchor="w", text_color="#8B008B").pack(side="left", padx=5)
        self.name_entry = ctk.CTkEntry(name_row, placeholder_text="Напр. Samsung WW80", fg_color="#FFFFFF", text_color="#000000")
        self.name_entry.pack(side="right", padx=5, fill="x", expand=True)

    def add_manually(self):
        if self.current_mode != "calc":
            return
        try:
            name = self.name_entry.get().strip()
            if not name:
                raise ValueError("Не указано название образца")
            if len(name) > 50:
                raise ValueError("Название слишком длинное (макс. 50 символов)")

            internal_type = TYPE_MAPPING[self.type_var.get()]
            characteristics = {}

            for param, entry in self.entries.items():
                val = entry.get().strip()
                if not val:
                    raise ValueError(f"Не заполнен параметр: {param}")
                try:
                    num = float(val)
                except ValueError:
                    raise ValueError(f"Значение '{val}' для {param} не является числом")
                if num <= 0:
                    raise ValueError(f"{param} должно быть положительным числом")
                characteristics[param] = num

            new_appliance = Appliance(name=name, appliance_type=internal_type, characteristics=characteristics)
            self.appliances.append(new_appliance)
            self.update_list_display()
            messagebox.showinfo("Успех", f"Образец '{name}' добавлен.")

            for entry in self.entries.values():
                entry.delete(0, "end")
            self.name_entry.delete(0, "end")

        except ValueError as e:
            messagebox.showerror("Ошибка ввода", str(e))
        except Exception as e:
            messagebox.showerror("Неожиданная ошибка", str(e))

    def load_from_file(self):
        if self.current_mode != "calc":
            return
        filepath = filedialog.askopenfilename(
            title="Выберите файл данных",
            filetypes=[("Data Files", "*.json *.csv *.txt"), ("All Files", "*.*")]
        )
        if not filepath:
            return

        ext = filepath.split(".")[-1].lower()
        internal_type = TYPE_MAPPING[self.type_var.get()]
        loaded = []

        try:
            if ext == "json":
                loaded = _parse_json(filepath, internal_type)
            elif ext == "csv":
                loaded = _parse_csv(filepath, internal_type)
            elif ext == "txt":
                loaded = _parse_txt(filepath, internal_type)
            else:
                messagebox.showwarning("Формат", "Поддерживаются только .json, .csv, .txt")
                return

            if loaded:
                self.appliances.extend(loaded)
                self.update_list_display()
                messagebox.showinfo("Загрузка", f"Успешно загружено: {len(loaded)} образцов.")
            else:
                messagebox.showinfo("Загрузка", "Валидные данные не найдены или файл пуст.")
        except Exception as e:
            messagebox.showerror("Ошибка файла", str(e))

    def update_list_display(self):
        self.list_text.delete("1.0", "end")
        if not self.appliances:
            self.list_text.insert("end", "Список пуст")
            return
        for i, app in enumerate(self.appliances, 1):
            self.list_text.insert("end", f"{i}. [{app.type}] {app.name}\n")

    def clear_list(self):
        self.appliances.clear()
        self.update_list_display()

    def run_calculation(self):
        if self.current_mode != "calc":
            return
        if len(self.appliances) < 1:
            messagebox.showwarning("Нет данных", "Добавьте или загрузите хотя бы один образец.")
            return

        internal_type = TYPE_MAPPING[self.type_var.get()]
        weights = WEIGHTS[internal_type]

        try:
            normalized = normalize_relative_to_first(self.appliances, internal_type)
            tech_levels = calculate_weighted_tech_level(normalized, weights)

            models = list(normalized.keys())
            all_params = list(normalized[models[0]].keys())
            values = [[normalized[m][p] for p in all_params] for m in models]

            create_radial(models, all_params, values)
            plot_tech_levels_bar(tech_levels, title=f"Технический уровень: {self.type_var.get()}")
            messagebox.showinfo("Готово", "Графики успешно построены.")
        except Exception as e:
            messagebox.showerror("Ошибка расчёта", str(e))

    def run_bubble_chart(self):
        if self.current_mode != "calc":
            return
        if len(self.appliances) < 2:
            messagebox.showwarning("Нет данных", "Нужно минимум 2 образца для пузырьковой диаграммы.")
            return
        internal_type = TYPE_MAPPING[self.type_var.get()]
        weights = WEIGHTS[internal_type]
        try:
            normalized = normalize_relative_to_first(self.appliances, internal_type)
            tech_levels = calculate_weighted_tech_level(normalized, weights)
            plot_bubble_chart(self.appliances, tech_levels, internal_type)
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def load_production_data(self):
        filepath = filedialog.askopenfilename(
            title="Выберите CSV файл с данными для прогнозирования",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not filepath:
            return
        try:
            df = pd.read_csv(filepath, delimiter=',')
            required_columns = self.forecast_features + [self.forecast_target]
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"В файле отсутствует колонка {col}")

            original_len = len(df)
            df = df[(df['steel_price'] > 0) &
                    (df['plastic_price'] > 0) &
                    (df['is_pre_holiday_season'].isin([0, 1])) &
                    (df['production_capacity_utilization'] >= 0) &
                    (df['production_capacity_utilization'] <= 100) &
                    (df['avg_worker_skill_level'] >= 1) &
                    (df['avg_worker_skill_level'] <= 10) &
                    (df['monthly_appliance_production'] > 0)]

            filtered_len = len(df)
            if filtered_len == 0:
                raise ValueError("После фильтрации аномалий не осталось ни одной строки. Проверьте данные.")
            if filtered_len < original_len:
                messagebox.showwarning("Фильтрация данных", f"Удалено {original_len - filtered_len} строк с аномалиями.\nОсталось {filtered_len} строк.")

            X = df[self.forecast_features]
            y = df[self.forecast_target]
            model = LinearRegression()
            model.fit(X, y)
            self.forecast_model = model
            self.forecast_data = df
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            self.forecast_status.configure(text=f"Модель обучена. R² = {r2:.3f}, MSE = {mse:.2f}", text_color="#2b8a3e")
            messagebox.showinfo("Успех", f"Данные загружены, модель обучена.\nИсходных строк: {original_len}\nПосле очистки: {filtered_len}\nR² = {r2:.3f}\nMSE = {mse:.2f}")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
            self.forecast_status.configure(text="Ошибка загрузки данных", text_color="red")

    def predict_production(self):
        if self.forecast_model is None:
            messagebox.showwarning("Нет модели", "Сначала загрузите данные и обучите модель.")
            return
        try:
            input_values = []
            for name in self.forecast_features:
                val_str = self.feature_entries[name].get().strip()
                if not val_str:
                    raise ValueError(f"Не введено значение для {name}")
                val = float(val_str)
                if name == 'is_pre_holiday_season' and val not in (0, 1):
                    raise ValueError("Предпраздничный сезон должен быть 0 или 1")
                if name == 'production_capacity_utilization' and (val < 0 or val > 100):
                    raise ValueError("Загрузка мощностей от 0 до 100")
                if name == 'avg_worker_skill_level' and (val < 1 or val > 10):
                    raise ValueError("Уровень квалификации от 1 до 10")
                if name in ('steel_price', 'plastic_price') and val <= 0:
                    raise ValueError(f"{name} должна быть положительной")
                input_values.append(val)
            input_df = pd.DataFrame([input_values], columns=self.forecast_features)
            prediction = self.forecast_model.predict(input_df)[0]
            self.prediction_label.configure(text=f"Прогнозируемый объём производства: {prediction:.2f} тыс. шт.")
        except Exception as e:
            messagebox.showerror("Ошибка прогноза", str(e))

    def plot_dependence(self):
        if self.forecast_model is None:
            messagebox.showwarning("Нет модели", "Сначала загрузите данные и обучите модель.")
            return
        try:
            feature = self.interval_feature.get()
            min_val = float(self.min_entry.get())
            max_val = float(self.max_entry.get())
            if min_val >= max_val:
                raise ValueError("Минимум должен быть меньше максимума")
            points = 50
            x_vals = np.linspace(min_val, max_val, points)
            fixed_values = {}
            for name in self.forecast_features:
                if name == feature:
                    continue
                val_str = self.feature_entries[name].get().strip()
                if not val_str:
                    raise ValueError(f"Не введено значение для {name} (фиксируется)")
                val = float(val_str)
                if name == 'production_capacity_utilization' and (val < 0 or val > 100):
                    raise ValueError("Загрузка мощностей от 0 до 100")
                if name == 'avg_worker_skill_level' and (val < 1 or val > 10):
                    raise ValueError("Уровень квалификации от 1 до 10")
                if name in ('steel_price', 'plastic_price') and val <= 0:
                    raise ValueError(f"{name} должна быть положительной")
                fixed_values[name] = val
            X_grid = []
            for x in x_vals:
                row = []
                for name in self.forecast_features:
                    if name == feature:
                        row.append(x)
                    else:
                        row.append(fixed_values[name])
                X_grid.append(row)
            X_grid_df = pd.DataFrame(X_grid, columns=self.forecast_features)
            y_pred = self.forecast_model.predict(X_grid_df)
            plt.figure(figsize=(8, 5))
            plt.plot(x_vals, y_pred, color='#FF69B4', linewidth=2)
            plt.xlabel(feature, fontsize=12)
            plt.ylabel("Прогноз производства (тыс. шт)", fontsize=12)
            plt.title(f"Зависимость объёма производства от {feature}", fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            messagebox.showerror("Ошибка построения графика", str(e))

if __name__ == "__main__":
    app = TechLevelApp()
    app.mainloop()