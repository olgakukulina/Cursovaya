import customtkinter as ctk
from tkinter import messagebox, filedialog
import matplotlib.pyplot as plt
import os

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


class TechLevelApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Оценка технического уровня бытовой техники")
        self.geometry("880x720")
        self.appliances = []  # Список объектов Appliance
        self.entries = {}  # Словарь для хранения виджетов ввода
        self.type_var = ctk.StringVar(value="Стиральные машины")

        self.setup_ui()

    def setup_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        top_frame = ctk.CTkFrame(self)
        top_frame.grid(row=0, column=0, columnspan=2, padx=20, pady=10, sticky="ew")

        ctk.CTkLabel(top_frame, text="Тип техники:").pack(side="left", padx=5)
        self.type_menu = ctk.CTkOptionMenu(top_frame, values=list(TYPE_MAPPING.keys()), variable=self.type_var,
                                           command=self.on_type_change)
        self.type_menu.pack(side="left", padx=5)

        ctk.CTkButton(top_frame, text="Загрузить из файла", command=self.load_from_file).pack(side="right", padx=5)
        ctk.CTkButton(top_frame, text="Добавить вручную", command=self.add_manually).pack(side="right", padx=5)
        self.input_frame = ctk.CTkFrame(self)
        self.input_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        self.build_input_form("Стиральные машины")
        self.list_frame = ctk.CTkFrame(self)
        self.list_frame.grid(row=1, column=1, padx=20, pady=10, sticky="nsew")
        ctk.CTkLabel(self.list_frame, text="Загруженные образцы:", font=ctk.CTkFont(weight="bold")).pack(pady=5)

        self.list_text = ctk.CTkTextbox(self.list_frame, width=350)
        self.list_text.pack(padx=10, pady=5, fill="both", expand=True)
        ctk.CTkButton(self.list_frame, text="Очистить список", command=self.clear_list).pack(pady=5)
        calc_frame = ctk.CTkFrame(self)
        calc_frame.grid(row=2, column=0, columnspan=2, padx=20, pady=10, sticky="ew")
        ctk.CTkButton(calc_frame, text="Рассчитать ТУ и построить графики", fg_color="#2b8a3e", hover_color="#1c6f30",
                      command=self.run_calculation).pack(fill="x")

    def on_type_change(self, choice):
        self.build_input_form(choice)

    def build_input_form(self, type_name: str):
        for widget in self.input_frame.winfo_children():
            widget.destroy()
        self.entries.clear()

        ctk.CTkLabel(self.input_frame, text=f"Ввод параметров: {type_name}", font=ctk.CTkFont(weight="bold")).pack(
            pady=5)

        internal_type = TYPE_MAPPING[type_name]
        params = PARAMS_MAP[internal_type]

        for i, param in enumerate(params):
            row = ctk.CTkFrame(self.input_frame)
            row.pack(fill="x", padx=10, pady=2)

            ctk.CTkLabel(row, text=f"{param}:", width=200, anchor="w").pack(side="left", padx=5)
            entry = ctk.CTkEntry(row, placeholder_text="Введите число")
            entry.pack(side="right", padx=5, fill="x", expand=True)
            self.entries[param] = entry

        name_row = ctk.CTkFrame(self.input_frame)
        name_row.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(name_row, text="Название образца:", width=200, anchor="w").pack(side="left", padx=5)
        self.name_entry = ctk.CTkEntry(name_row, placeholder_text="Напр. Samsung WW80")
        self.name_entry.pack(side="right", padx=5, fill="x", expand=True)

    def add_manually(self):
        try:
            name = self.name_entry.get().strip()
            if not name:
                raise ValueError("Не указано название образца")

            internal_type = TYPE_MAPPING[self.type_var.get()]
            characteristics = {}

            for param, entry in self.entries.items():
                val = entry.get().strip()
                if not val:
                    raise ValueError(f"Не заполнен параметр: {param}")
                characteristics[param] = float(val)

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
        if len(self.appliances) < 1:
            messagebox.showwarning("Нет данных", "Добавьте или загрузите хотя бы один образец.")
            return

        internal_type = TYPE_MAPPING[self.type_var.get()]

        try:
            normalized = normalize_for_radar(self.appliances, internal_type)
            tech_levels = calculate_tech_level_simple(normalized)
            create_radial(
                models=list(normalized.keys()),
                params=list(normalized[next(iter(normalized))].keys()),
                values=[list(v.values()) for v in normalized.values()]
            )
            plot_tech_levels_bar(tech_levels, title=f"Технический уровень: {self.type_var.get()}")
            messagebox.showinfo("Готово", "Графики успешно построены.")
        except Exception as e:
            messagebox.showerror("Ошибка расчёта", str(e))


if __name__ == "__main__":
    app = TechLevelApp()
    app.mainloop()