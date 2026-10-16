"""Visible infrastructure fixtures only. Never enters a real-data manifest."""
from PIL import Image, ImageDraw


def shapes():
    rows = []
    colors = ["red", "blue", "green", "yellow"]
    for i in range(12):
        image = Image.new("RGB", (384,256), "white")
        draw = ImageDraw.Draw(image)
        color = colors[i % 4]
        draw.rectangle((40,50,150,180), fill=color)
        draw.ellipse((220,60,330,170), fill=colors[(i+1)%4])
        # Change shape position slightly without changing question interpretation.
        draw.line((0,240-i,384,240-i), fill="gray", width=1)
        rows.append({"id":f"synthetic-{i}", "image":image,
                     "question":"What color is the rectangle?", "options":colors.copy(), "gold":"ABCD"[i%4]})
    return rows
