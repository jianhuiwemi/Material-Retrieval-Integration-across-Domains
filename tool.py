import cv2
import numpy as np
import os
import tkinter as tk
from PIL import Image, ImageTk

image_folder = "path"
image_path = os.path.join(image_folder, "fig.png")
mask_path = os.path.join(image_folder, "mask.png")

image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image file not found: {image_path}")
original_image = image.copy()
height, width = image.shape[:2]

mask = np.zeros((height, width), dtype=np.uint8)

INITIAL_CIRCLE_RADIUS = 100

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
pil_image = Image.fromarray(image_rgb)

root = tk.Tk()
root.title("Image Annotation Tool")

canvas_frame = tk.Frame(root)
canvas_frame.pack(fill=tk.BOTH, expand=True)

h_scroll = tk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL)
h_scroll.pack(side=tk.BOTTOM, fill=tk.X)

v_scroll = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL)
v_scroll.pack(side=tk.RIGHT, fill=tk.Y)

canvas = tk.Canvas(
    canvas_frame, width=800, height=600,
    scrollregion=(0, 0, width, height),
    xscrollcommand=h_scroll.set,
    yscrollcommand=v_scroll.set
)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

h_scroll.config(command=canvas.xview)
v_scroll.config(command=canvas.yview)

tk_image = ImageTk.PhotoImage(pil_image)
canvas_image = canvas.create_image(0, 0, anchor="nw", image=tk_image)

circles = []

current_radius = tk.IntVar(value=INITIAL_CIRCLE_RADIUS)

def on_click(event):
    x = canvas.canvasx(event.x)
    y = canvas.canvasy(event.y)
    x = int(x)
    y = int(y)
    radius = current_radius.get()
    print(f"Clicked at: ({x}, {y}), Radius: {radius}")
    cv2.circle(mask, (x, y), radius, 255, -1)
    canvas.create_oval(x - radius, y - radius, x + radius, y + radius, outline="red", width=2)
    circles.append((x, y, radius))

def save_mask():
    cv2.imwrite(mask_path, mask)
    print(f"Mask saved to: {mask_path}")
    masked_image = cv2.bitwise_and(original_image, original_image, mask=mask)
    masked_image_path = os.path.join(image_folder, "masked_fig.png")
    cv2.imwrite(masked_image_path, masked_image)
    print(f"Masked image saved to: {masked_image_path}")

def clear_canvas():
    global mask, circles
    mask = np.zeros((height, width), dtype=np.uint8)
    circles = []
    canvas.delete("all")
    canvas.create_image(0, 0, anchor="nw", image=tk_image)

canvas.bind("<Button-1>", on_click)

button_frame = tk.Frame(root)
button_frame.pack(fill=tk.X, padx=5, pady=5)

save_button = tk.Button(button_frame, text="Save Mask (S)", command=save_mask)
save_button.pack(side=tk.LEFT, padx=5, pady=5)

clear_button = tk.Button(button_frame, text="Clear Drawings (C)", command=clear_canvas)
clear_button.pack(side=tk.LEFT, padx=5, pady=5)

radius_frame = tk.Frame(button_frame)
radius_frame.pack(side=tk.LEFT, padx=20)

radius_label = tk.Label(radius_frame, text="Circle Radius:")
radius_label.pack(side=tk.LEFT)

radius_slider = tk.Scale(radius_frame, from_=10, to=300, orient=tk.HORIZONTAL, variable=current_radius)
radius_slider.pack(side=tk.LEFT)

radius_value_label = tk.Label(radius_frame, textvariable=current_radius)
radius_value_label.pack(side=tk.LEFT, padx=5)

def on_key_press(event):
    if event.char.lower() == 's':
        save_mask()
    elif event.char.lower() == 'c':
        clear_canvas()
    elif event.char.lower() == 'q':
        root.quit()

root.bind("<Key>", on_key_press)

print("Click on the image to mark regions of interest. Press 'S' to save mask, 'C' to clear, and 'Q' to quit.")

root.mainloop()
