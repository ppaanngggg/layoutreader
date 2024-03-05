import cv2
import fitz
import requests

# Please `python main.py` first

doc = fitz.open("./test.pdf")
for i, page in enumerate(doc):
    page_img_file = f"./page_{i}.png"
    pix = page.get_pixmap()
    pix.save(page_img_file)
    dicts = page.get_text("dict")
    # get width, height and boxes
    width = dicts["width"]
    height = dicts["height"]
    boxes = []
    for block in dicts["blocks"]:
        if "lines" not in block:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                boxes.append(span["bbox"])
    # send to server to predict orders
    r = requests.post(
        "http://localhost:8000/predict",
        json={"boxes": boxes, "width": width, "height": height},
    )
    orders = r.json()["orders"]
    # reorder boxes
    boxes = [boxes[i] for i in orders]
    # draw boxes
    img = cv2.imread(page_img_file)
    for idx, box in enumerate(boxes):
        x0, y0, x1, y1 = box
        x0 = round(x0)
        y0 = round(y0)
        x1 = round(x1)
        y1 = round(y1)
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 1)
        cv2.putText(
            img,
            str(idx),
            (x1, y1),
            cv2.FONT_HERSHEY_PLAIN,
            0.5,
            (0, 0, 255),
            1,
        )
    cv2.imwrite(page_img_file, img)
