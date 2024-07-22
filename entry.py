import tkinter as tk
from tkinter import messagebox, Button, Frame, colorchooser
from PIL import Image, ImageOps, ImageDraw
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf

class PaintApp:
    def __init__(self, root):
        self.root = root
        self.root.geometry("1000x800")

        self.brush_color = "black"
        self.brush_size = 5
        self.canvas_width = 800
        self.canvas_height = 600
        self.canvas = tk.Canvas(root, bg="white", width=self.canvas_width, height=self.canvas_height)
        self.canvas.pack()

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)

        self.setup_controls()

        self.points = []

        self.model = self.load_model()

    def choose_color(self):
        color = colorchooser.askcolor()[1]
        if color:
            self.brush_color = color

    def change_brush_size(self, event):
        self.brush_size = event.widget.get()

    def setup_controls(self):
        controls_frame = Frame(self.root)
        controls_frame.pack()

        clear_button = Button(controls_frame, text="Clear", command=self.clear_canvas)
        clear_button.grid(row=0, column=0)

        color_button = tk.Button(controls_frame, text="Color", command=self.choose_color)
        color_button.grid(row=0, column=1)

        size_slider = tk.Scale(controls_frame, from_=5, to=10, orient=tk.HORIZONTAL, label="Brush Size")
        size_slider.set(self.brush_size)
        size_slider.grid(row=0, column=2)
        size_slider.bind("<Motion>", self.change_brush_size)

        self.predict_label = tk.Label(controls_frame, text="Nhãn dự đoán: ")
        self.predict_label.grid(row=0, column=4)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.points = []

    def paint(self, event):
        x1, y1 = event.x - self.brush_size, event.y - self.brush_size
        x2, y2 = event.x + self.brush_size, event.y + self.brush_size
        self.canvas.create_oval(x1, y1, x2, y2, fill=self.brush_color, outline=self.brush_color)
        self.points.append((x1, y1, x2, y2))

    def on_mouse_release(self, event):
        self.predict_digit()

    def prepare_input_image(self):
        pil_image = Image.new("L", (self.canvas_width, self.canvas_height), 255)
        draw = ImageDraw.Draw(pil_image)
        for x1, y1, x2, y2 in self.points:
            draw.ellipse([x1, y1, x2, y2], fill=self.brush_color, outline=self.brush_color)

        pil_image = pil_image.resize((150, 150))
        pil_image = ImageOps.invert(pil_image)
        pil_image = pil_image.convert('L')
        image_array = np.array(pil_image)
        image_array = image_array.reshape((1, 150, 150, 1))
        image_array = image_array / 255.0
        return image_array

    def load_model(self):
        model_path = 'D:/barovinh/Python/Dataset/model.h5'
        model = load_model(model_path)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    label_map = {
        0: 'Cối-82',
        1: 'Cờ sch-bchqs-quận huyện',
        2: 'Cờ vị trí chỉ huy đại đội',
        3: 'Cờ vị trí chỉ huy tiểu đoàn',
        4: 'Cờ vị trí chỉ huy trung đội',
        5: 'Cụm xe thiết giáp',
        6: 'Đại đội cối 106,7',
        7: 'Đánh phá',
        8: 'Điểm tựa phòng ngự',
        9: 'Hàng rào thép rai',
        10: 'Khu vực đổ bộ đường không',
        11: 'Kí hiệu xe tăng',
        12: 'Máy bay trực thăng vũ trang',
        13: 'Súng-DKZ-cò-75',
        14: 'Trận địa trung đội cối 82'
    }

    def predict_digit(self):
        input_image = self.prepare_input_image()
        predictions = self.model.predict(input_image)

        top_k = 3
        top_k_values, top_k_indices = tf.nn.top_k(predictions, k=top_k)

        top_k_indices = top_k_indices.numpy()[0]
        top_k_labels = [self.label_map.get(index, "Unknown") for index in top_k_indices]

        print(f"Top {top_k} Predicted Indices: {top_k_indices}")
        print(f"Top {top_k} Predicted Labels: {top_k_labels}")

        self.predict_label.config(text=f"Nhãn dự đoán: {', '.join(top_k_labels)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = PaintApp(root)
    root.mainloop()

