import customtkinter as ctk
from tkinter import messagebox, filedialog
import json
import csv
import numpy as np
import matplotlib.pyplot as plt

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
        self.current_mode = "calc"  # calc, forecast, optim

        self.setup_ui()

    def setup_ui(self):
        # Главный контейнер
        self.main_frame = ctk.CTkFrame(self, fg_color="#FFE4E1")
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=10)

        # Верхняя панель с переключателем режимов
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

        # Контейнер для содержимого режимов
        self.content_frame = ctk.CTkFrame(self.main_frame, fg_color="#FFE4E1")
        self.content_frame.pack(fill="both", expand=True)

        # Режим "Расчёт ТУ" – все рабочие виджеты
        self.calc_frame = ctk.CTkFrame(self.content_frame, fg_color="#FFE4E1")
        self.calc_frame.pack(fill="both", expand=True)

        # Создаём виджеты расчёта внутри calc_frame
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

        # Фрейм для формы ввода
        self.input_frame = ctk.CTkFrame(self.calc_frame, fg_color="#FFF0F5")
        self.input_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        self.build_input_form("Стиральные машины")

        # Фрейм для списка
        self.list_frame = ctk.CTkFrame(self.calc_frame, fg_color="#FFF0F5")
        self.list_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        ctk.CTkLabel(self.list_frame, text="Загруженные образцы:", font=ctk.CTkFont(weight="bold"), text_color="#8B008B").pack(pady=5)
        self.list_text = ctk.CTkTextbox(self.list_frame, width=350, fg_color="#FFFFFF", text_color="#000000")
        self.list_text.pack(padx=10, pady=5, fill="both", expand=True)
        ctk.CTkButton(self.list_frame, text="Очистить список", command=self.clear_list,
                      fg_color="#FF69B4", hover_color="#FF1493", text_color="white").pack(pady=5)

        # Кнопки расчёта
        calc_buttons = ctk.CTkFrame(self.calc_frame, fg_color="#FFE4E1")
        calc_buttons.pack(side="bottom", fill="x", padx=10, pady=10)

        ctk.CTkButton(calc_buttons, text="Рассчитать ТУ и построить графики",
                      fg_color="#2b8a3e", hover_color="#1c6f30", text_color="white",
                      command=self.run_calculation).pack(fill="x", pady=2)

        ctk.CTkButton(calc_buttons, text="Пузырьковая диаграмма",
                      fg_color="#FF69B4", hover_color="#FF1493", text_color="white",
                      command=self.run_bubble_chart).pack(fill="x", pady=2)

        # Режим "Прогнозирование" – заглушка
        self.forecast_frame = ctk.CTkFrame(self.content_frame, fg_color="#FFF0F5")
        forecast_label = ctk.CTkLabel(self.forecast_frame, text="Функция прогнозирования находится в разработке",
                                      font=ctk.CTkFont(size=20, weight="bold"), text_color="#8B008B")
        forecast_label.pack(expand=True)
        # Режим "Оптимизация" – заглушка
        self.optim_frame = ctk.CTkFrame(self.content_frame, fg_color="#FFF0F5")
        optim_label = ctk.CTkLabel(self.optim_frame, text="Функция оптимизации находится в разработке",
                                   font=ctk.CTkFont(size=20, weight="bold"), text_color="#8B008B")
        optim_label.pack(expand=True)

        # Показываем начальный режим
        self.change_mode("Расчёт ТУ")

    def change_mode(self, mode):
        """Переключает видимый фрейм в зависимости от выбранного режима"""
        # Скрываем все фреймы
        self.calc_frame.pack_forget()
        self.forecast_frame.pack_forget()
        self.optim_frame.pack_forget()

        if mode == "Расчёт ТУ":
            self.calc_frame.pack(fill="both", expand=True)
            self.current_mode = "calc"
        elif mode == "Прогнозирование":
            self.forecast_frame.pack(fill="both", expand=True)
            self.current_mode = "forecast"
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

if __name__ == "__main__":
    app = TechLevelApp()
    app.mainloop()