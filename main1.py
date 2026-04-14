import matplotlib.pyplot as plt
import numpy as np

laptops_by_char = {
    "Процессор (ГГц)": {
        "ASUS ROG Strix": 2.6,
        "Lenovo Legion 5": 3.0,
        "HP Omen 15": 2.8,
        "MSI GF65": 2.5
    },
    "ОЗУ (ГБ)": {
        "ASUS ROG Strix": 16,
        "Lenovo Legion 5": 32,
        "HP Omen 15": 16,
        "MSI GF65": 16
    },
    "SSD (ГБ)": {
        "ASUS ROG Strix": 512,
        "Lenovo Legion 5": 1000,
        "HP Omen 15": 512,
        "MSI GF65": 256
    },
    "Видеокарта (ГБ)": {
        "ASUS ROG Strix": 8,
        "Lenovo Legion 5": 6,
        "HP Omen 15": 4,
        "MSI GF65": 6
    },
    "Экран (дюймы)": {
        "ASUS ROG Strix": 15.6,
        "Lenovo Legion 5": 15.6,
        "HP Omen 15": 15.6,
        "MSI GF65": 15.6
    },
    "Вес (кг)": {
        "ASUS ROG Strix": 2.4,
        "Lenovo Legion 5": 2.7,
        "HP Omen 15": 2.3,
        "MSI GF65": 2.1
    }
}
print(isinstance(laptops_by_char, dict))
n = laptops_by_char.values()
models = list(list(laptops_by_char.values())[0].keys())
name_char = list(laptops_by_char.keys())
all_char = []
for char in n:
    all_char.append(list(char.values()))
char = [list(row) for row in zip(*all_char)]


def get_normal(char):
    normal = []
    for item in char:
        normal.append([a / b for a, b in zip(item, char[0])])
    return normal


def get_quality(normal):
    result = []
    for item in normal:
        result.append(round(sum(item) / len(item), 2))
    return result


def create_bar(name, values):
    plt.bar(name, values)
    plt.xlabel("Модель")
    plt.ylabel("Kту")
    plt.legend
    plt.show()


def create_radial(models, name, values):

    for item in values:
        item += item[:1]

    angles = np.linspace(0, 2 * np.pi, len(name), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))

    for i in range(len(values)):
        ax.plot(angles, values[i], "o-", linewidth=2, label=models[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(name, fontsize=10)
    ax.set_ylim(0, 2)

    # Легенда и заголовок
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
    plt.title("Сравнение относительных характеристик", pad=20)
    plt.show()


data = get_quality(get_normal(char))
create_bar(models, data)
create_radial(models, name_char, get_normal(char))

print(get_normal(char))