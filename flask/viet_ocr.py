import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import onnxruntime as ort
import cv2
from pyvi import ViTokenizer
from doctr.models.recognition.crnn import crnn_mobilenet_v3_large
import torch
from torchvision import transforms
import pyclipper

VOCAB = list("aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!\"#$%&''()*+,-./:;<=>?@[\\]^_`{|}~")
# --- Perspective transform from 4 points ---
def four_point_transform(image, pts):
    pts = np.array(pts, dtype="float32")
    (tl, tr, br, bl) = pts
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(np.array(image), M, (maxWidth, maxHeight))
    return Image.fromarray(warped)

# --- DBNet Detector ---
class DBNetDetector:
    def __init__(self, onnx_path, bin_thresh=0.3, box_thresh=0.1, unclip_ratio=1.5):
        self.session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        self.bin_thresh = bin_thresh
        self.box_thresh = box_thresh
        self.unclip_ratio = unclip_ratio

    def preprocess(self, pil_img):
        img = pil_img.convert('RGB')
        w, h = img.size
        target_size = 512
        ratio = w / h
        if ratio > 1:
            resized_w = target_size
            resized_h = int(target_size / ratio)
        else:
            resized_h = target_size
            resized_w = int(target_size * ratio)
        resized_w = max(1, resized_w)
        resized_h = max(1, resized_h)
        pad_left = (target_size - resized_w) // 2
        pad_top = (target_size - resized_h) // 2
        img_resized = img.resize((resized_w, resized_h), Image.BILINEAR)
        new_img = Image.new('RGB', (target_size, target_size), (0,0,0))
        new_img.paste(img_resized, (pad_left, pad_top))
        img_np = np.array(new_img).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = (img_np - mean) / std
        img_np = img_np.transpose(2, 0, 1)
        img_np = np.expand_dims(img_np, 0)
        print(f"[DBNet] preprocess: orig_size={w}x{h}, resized={resized_w}x{resized_h}, pad_left={pad_left}, pad_top={pad_top}")
        return img_np.astype(np.float32), (w, h), resized_w, resized_h, pad_left, pad_top

    def unclip_polygon(self, contour):
        area = cv2.contourArea(contour)
        length = cv2.arcLength(contour, True)
        if length == 0:
            return None
        distance = area * self.unclip_ratio / (length + 1e-6)
        poly = contour.squeeze().astype(np.int32)
        pc = pyclipper.PyclipperOffset()
        pc.AddPath(poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = pc.Execute(distance)
        if len(expanded) == 0:
            return None
        return np.array(expanded[0]).reshape(-1, 2)

    def box_score(self, prob_map, polygon):
        mask = np.zeros(prob_map.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [polygon.astype(np.int32)], 1)
        return float(np.mean(prob_map[mask == 1]))

    def postprocess(self, pred, orig_size, resized_w, resized_h, pad_left, pad_top):
        pred = pred[0, 0]
        # Apply sigmoid
        prob_map = 1 / (1 + np.exp(-pred))
        # Binarize
        mask = prob_map > self.bin_thresh
        mask = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        h, w = prob_map.shape
        scale_w = orig_size[0] / resized_w
        scale_h = orig_size[1] / resized_h
        for cnt in contours:
            if cv2.contourArea(cnt) < 2:
                continue
            # --- Unclip/expand contour bằng pyclipper ---
            try:
                cnt_expanded = self.unclip_polygon(cnt)
            except Exception:
                cnt_expanded = None
            if cnt_expanded is None or len(cnt_expanded) < 4:
                continue
            rect = cv2.minAreaRect(cnt_expanded)
            box = cv2.boxPoints(rect)
            box = np.array(box)
            # Calculate confidence score giống Doctr
            score = self.box_score(prob_map, cnt_expanded)
            if score < self.box_thresh:
                continue
            # Scale box về ảnh gốc
            box[:,0] = (box[:,0] - pad_left) * scale_w
            box[:,1] = (box[:,1] - pad_top) * scale_h
            boxes.append(box.astype(np.int32))
        return boxes

    def detect(self, pil_img):
        img_np, orig_size, resized_w, resized_h, pad_left, pad_top = self.preprocess(pil_img)
        ort_inputs = {self.session.get_inputs()[0].name: img_np}
        pred = self.session.run(None, ort_inputs)[0]
        boxes = self.postprocess(pred, orig_size, resized_w, resized_h, pad_left, pad_top)
        return boxes

# --- Post-process Vietnamese text ---
def post_process_text(text: str) -> str:
    try:
        tokenized_text = ViTokenizer.tokenize(text)
        readable_text = tokenized_text.replace('_', ' ')
        return readable_text
    except Exception:
        return text

# --- PyTorch CRNN Recognizer ---
crnn_model = crnn_mobilenet_v3_large(pretrained=False, vocab=VOCAB)
# crnn_model = crnn_model.cuda()
checkpoint = torch.load("best_checkpoint_printed3.pth", map_location="cpu")
crnn_model.load_state_dict(checkpoint["model_state_dict"])
crnn_model.eval()

crnn_transform = transforms.Compose([
    transforms.Resize((32, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def recognize_pytorch(img_pil):
    img = crnn_transform(img_pil.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        out = crnn_model(img)
    preds = out["preds"]
    text, conf = preds[0]
    return text

# --- Instantiate models ---
dbnet_detector = DBNetDetector("db_mobilenet_v3_large.onnx")

# --- Main pipeline ---
def ocr_pipeline(img):
    if img is None:
        return "Vui lòng upload ảnh!", None
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    print(f"[OCR] Input image size: {img.size}")
    boxes = dbnet_detector.detect(img)
    print(f"[OCR] Number of detected boxes: {len(boxes)}")
    results = []
    vis_img = img.copy().convert("RGB")
    draw = ImageDraw.Draw(vis_img)
    for box in boxes:
        xmin = int(np.min(box[:,0]))
        xmax = int(np.max(box[:,0]))
        ymin = int(np.min(box[:,1]))
        ymax = int(np.max(box[:,1]))
        print(f"[OCR] Crop box: ({xmin},{ymin})-({xmax},{ymax})")
        if xmax-xmin<5 or ymax-ymin<5:
            continue
        # --- Expand box by 10% each side ---
        w = xmax - xmin
        h = ymax - ymin
        expand_w = int(w * 0.1)
        expand_h = int(h * 0.1)
        xmin_exp = max(0, xmin - expand_w)
        ymin_exp = max(0, ymin - expand_h)
        xmax_exp = min(img.width, xmax + expand_w)
        ymax_exp = min(img.height, ymax + expand_h)
        crop = img.crop((xmin_exp, ymin_exp, xmax_exp, ymax_exp))
        text = recognize_pytorch(crop)
        if text.strip():
            results.append((box, text))
            draw.polygon([tuple(pt) for pt in box], outline="blue")
            try:
                font = ImageFont.truetype("arial.ttf", 15)
            except:
                font = ImageFont.load_default()
            draw.text((xmin, ymin-15 if ymin>15 else ymin), text, fill="blue", font=font)
    out_text = '\n'.join([t for _,t in results])
    post_text = post_process_text(out_text)
    print(f"[OCR] Final recognized text: {post_text}")
    return post_text, vis_img

# --- Gradio interface ---
demo = gr.Interface(
    fn=ocr_pipeline,
    inputs=gr.Image(label="Upload ảnh tài liệu"),
    outputs=[
        gr.Textbox(label="Kết quả nhận diện (word-level, đã word segment)", lines=10),
        gr.Image(type="pil", label="Ảnh kết quả (box + text)")
    ],
    title="Vietnamese OCR Pipeline (DBNet+CRNN ONNX)",
    description="1. Upload ảnh. 2. Detect word-level box bằng DBNet ONNX. 3. Recognize từng box bằng CRNN ONNX."
)

if __name__ == "__main__":
    demo.launch(debug=True, server_name="0.0.0.0", server_port=7862) 