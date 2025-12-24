import numpy as np
from numpy.linalg import norm
from scipy.optimize import linear_sum_assignment
from sentence_transformers import SentenceTransformer

# -------------------------------------
#  Sentence Embedding Model
# -------------------------------------
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def embed(text):
    return model.encode([text])[0]


def cosine_sim(a, b):
    return np.dot(a, b) / (norm(a) * norm(b) + 1e-9)


# -------------------------------------
#  IoU / IoA
# -------------------------------------
def iou(boxA, boxB):
    Ax1, Ay1, Ax2, Ay2 = boxA
    Bx1, By1, Bx2, By2 = boxB

    inter_x1 = max(Ax1, Bx1)
    inter_y1 = max(Ay1, By1)
    inter_x2 = min(Ax2, Bx2)
    inter_y2 = min(Ay2, By2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)

    inter_area = inter_w * inter_h
    areaA = (Ax2 - Ax1) * (Ay2 - Ay1)
    areaB = (Bx2 - Bx1) * (By2 - By1)

    union = areaA + areaB - inter_area
    return inter_area / union if union > 0 else 0.0


def ioa(boxA, boxB):
    Ax1, Ay1, Ax2, Ay2 = boxA
    Bx1, By1, Bx2, By2 = boxB

    inter_x1 = max(Ax1, Bx1)
    inter_y1 = max(Ay1, By1)
    inter_x2 = min(Ax2, Bx2)
    inter_y2 = min(Ay2, By2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)

    inter_area = inter_w * inter_h
    areaA = (Ax2 - Ax1) * (Ay2 - Ay1)
    return inter_area / areaA if areaA > 0 else 0.0


# -------------------------------------
#  Label similarity (embedding)
# -------------------------------------
def label_similarity(pred_label, gt_label):
    vp = embed(pred_label)
    vg = embed(gt_label)
    return cosine_sim(vp, vg)  # -1～1 → 負なら0 で後処理


# -------------------------------------
#  ① 最大 IoU（ラベル無視）
# -------------------------------------
def score_max_iou(pred, labels):
    total = 0.0
    for gt in labels:
        gt_box = gt["bbox_2d"]
        best = 0.0
        for p in pred:
            best = max(best, iou(p["bbox_2d"], gt_box))
        total += best
    return total


def score_max_ioa(pred, labels):
    total = 0.0
    for gt in labels:
        gt_box = gt["bbox_2d"]
        best = 0.0
        for p in pred:
            best = max(best, ioa(p["bbox_2d"], gt_box))
        total += best
    return total


def unused_ratio(pred, labels):
    # IoU が 0 の予測を数えるためのフラグ
    used = [False] * len(pred)
    if not used:
        return 0.0

    for gt in labels:
        gt_box = gt["bbox_2d"]

        best = 0.0
        best_idx = None

        # ベスト IoU を探して、その pred が使われたことを記録
        for i, p in enumerate(pred):
            iou_val = iou(p["bbox_2d"], gt_box)
            if iou_val > best:
                best_idx = i

        if best_idx is not None:
            used[best_idx] = True

    return used.count(False) / len(used)


def duplicate_iou(pred):
    total = 0.0
    n = len(pred)

    for i in range(n):
        for j in range(i + 1, n):
            box_i = pred[i]["bbox_2d"]
            box_j = pred[j]["bbox_2d"]

            total += iou(box_i, box_j)

    return total


# -------------------------------------
#  ② IoU × ラベル類似度
# -------------------------------------
def score_iou_with_label_sim(pred, labels):
    total = 0.0
    for gt in labels:
        gt_box = gt["bbox_2d"]
        gt_label = gt["label"]
        best = 0.0
        for p in pred:
            inter = iou(p["bbox_2d"], gt_box)
            sim = max(0.0, label_similarity(p["label"], gt_label))
            best = max(best, inter * sim)
        total += best
    return total


# -------------------------------------
# ③ IoA × ラベル類似度
# -------------------------------------
def score_ioa_with_label_sim(pred, labels):
    total = 0.0
    for gt in labels:
        gt_box = gt["bbox_2d"]
        gt_label = gt["label"]
        best = 0.0
        for p in pred:
            inter = ioa(p["bbox_2d"], gt_box)
            sim = max(0.0, label_similarity(p["label"], gt_label))
            best = max(best, inter * sim)
        total += best
    return total


# -------------------------------------
# ④ ハンガリアン法（IoU + ラベル類似度）
# -------------------------------------
def combined_score(pred_item, gt_item, alpha=1.0, beta=0.5):
    i = iou(pred_item["bbox_2d"], gt_item["bbox_2d"])
    s = max(0.0, label_similarity(pred_item["label"], gt_item["label"]))
    return alpha * i + beta * s


def optimal_matching_score(pred, labels, alpha=1.0, beta=0.5):
    if not pred or not labels:
        return 0.0

    cost = np.zeros((len(labels), len(pred)))
    for i, gt in enumerate(labels):
        for j, p in enumerate(pred):
            cost[i, j] = -combined_score(p, gt, alpha, beta)

    row_ind, col_ind = linear_sum_assignment(cost)
    return sum(-cost[r, c] for r, c in zip(row_ind, col_ind))


# -------------------------------------
#  動作テスト
# -------------------------------------
if __name__ == "__main__":
    pred = [
        {"bbox_2d": [0, 0, 1000, 1000], "label": "Drink"},
        {"bbox_2d": [0, 0, 1000, 1000], "label": "Drink"},
    ]

    labels = [
        {"bbox_2d": [106, 200, 200, 300], "label": "Drink in a bottle"},
        {"bbox_2d": [106, 200, 200, 300], "label": "Drink in a bottle"},
    ]

    print("max IoU:", score_max_iou(pred, labels))
    print("max IoA:", score_max_ioa(pred, labels))
    print("IoU × label_sim:", score_iou_with_label_sim(pred, labels))
    print("IoA × label_sim:", score_ioa_with_label_sim(pred, labels))
    print("optimal matching:", optimal_matching_score(pred, labels))
