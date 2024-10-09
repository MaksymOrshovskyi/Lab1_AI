from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw, ImageOps
import numpy as np


class ImageSegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Segmentation")

        # Створюємо елементи інтерфейсу
        self.load_button = Button(root, text="Load Image", command=self.load_image)
        self.load_button.pack()

        self.threshold_label = Label(root, text="Threshold (0-255):")
        self.threshold_label.pack()
        self.threshold_entry = Entry(root)
        self.threshold_entry.pack()
        self.threshold_entry.insert(0, "128")

        self.segments_label = Label(root, text="Number of Segments:")
        self.segments_label.pack()
        self.segments_entry = Entry(root)
        self.segments_entry.pack()
        self.segments_entry.insert(0, "10")

        self.canvas = Canvas(root, width=300, height=300)
        self.canvas.pack()

        self.image = None

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            image = Image.open(file_path)
            self.process_image(image)

    def process_image(self, image):
        # Змінюємо розмір зображення до 300x300 пікселів з використанням LANCZOS
        image = image.resize((300, 300), Image.Resampling.LANCZOS)

        # Перетворення зображення на чорно-біле з порогом
        threshold_value = int(self.threshold_entry.get())
        image_bw = image.convert("L")
        image_bw = ImageOps.invert(image_bw)
        image_bw = image_bw.point(lambda p: p > threshold_value and 255)

        # Показуємо сегментацію на зображенні
        num_segments = int(self.segments_entry.get())
        segmented_image = self.draw_segments(image_bw, num_segments)

        # Відображаємо зображення на Canvas
        self.image = ImageTk.PhotoImage(segmented_image)
        self.canvas.create_image(0, 0, anchor=NW, image=self.image)

        # Підрахунок чорних пікселів у кожному сегменті
        absolute_feature_vector = self.calculate_absolute_feature_vector(image_bw, num_segments)
        print("Absolute Feature Vector:", absolute_feature_vector)

        # Нормалізація за M1
        orshocskyi_m1 = self.normalize_s1(absolute_feature_vector)
        print("Оршовський (M1):", orshocskyi_m1)

        # Нормалізація за S1
        orshocskyi_s1 = self.normalize_m1(absolute_feature_vector)
        print("Оршовський (S1):", orshocskyi_s1)

    def draw_segments(self, image, num_segments):
        """Малюємо вертикальні лінії для візуалізації сегментації."""
        draw = ImageDraw.Draw(image)
        width, height = image.size
        segment_width = width // num_segments

        # Малюємо лінії
        for i in range(1, num_segments):
            x = i * segment_width
            draw.line([(x, 0), (x, height)], fill="red", width=2)

        return image

    def calculate_absolute_feature_vector(self, image, num_segments):
        """Розраховуємо абсолютний вектор ознак (кількість чорних пікселів у кожному сегменті)."""
        width, height = image.size
        segment_width = width // num_segments

        feature_vector = []
        for i in range(num_segments):
            # Вибираємо відповідний сегмент зображення
            segment = image.crop((i * segment_width, 0, (i + 1) * segment_width, height))
            # Підраховуємо кількість чорних пікселів (значення 0)
            black_pixels = np.sum(np.array(segment) == 0)
            feature_vector.append(np.int64(black_pixels))

        return feature_vector

    def normalize_s1(self, feature_vector):
        """Нормалізація вектора за S1 (за максимальним значенням)."""
        max_value = max(feature_vector)
        normalized_s1 = [np.float64(x / max_value) for x in feature_vector]
        return normalized_s1

    def normalize_m1(self, feature_vector):
        """Нормалізація вектора за M1 (за сумою всіх значень)."""
        total_sum = sum(feature_vector)
        normalized_m1 = [np.float64(x / total_sum) for x in feature_vector]
        return normalized_m1


# Створюємо та запускаємо вікно програми
root = Tk()
app = ImageSegmentationApp(root)
root.mainloop()
