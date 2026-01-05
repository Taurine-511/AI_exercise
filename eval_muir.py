import json
import re
import argparse
from typing import Any, Dict, Iterable, List, Optional, Tuple


def iter_generations(root: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(root, dict):
        gens = root.get("generations")
        if isinstance(gens, list):
            for g in gens:
                if isinstance(g, dict):
                    yield g
        return

    if isinstance(root, list):
        for item in root:
            if isinstance(item, dict) and isinstance(item.get("generations"), list):
                for g in item["generations"]:
                    if isinstance(g, dict):
                        yield g
        return


def normalize_ans(s: str) -> str:
    s = s.strip()
    # 余計な括弧や記号を軽く落とす（例: "(A)" "A." "Answer: A" など）
    s = re.sub(r"^[\(\[\{<\s]*", "", s)
    s = re.sub(r"[\)\]\}>\s\.\:：、。]+$", "", s)
    return s.strip().upper()


def extract_answer(text: str, valid_labels: Optional[set] = None) -> Tuple[Optional[str], str]:
    """
    返り値: (抽出結果 or None, どのルールで取れたかのタグ)
    """
    if text is None:
        return None, "none_text"

    t = str(text)

    # 1 <answer>...</answer> を最優先（複数あったら最後を採用）
    m_all = re.findall(r"<\s*answer\s*>\s*(.*?)\s*<\s*/\s*answer\s*>", t, flags=re.IGNORECASE | re.DOTALL)
    if m_all:
        cand = normalize_ans(m_all[-1])
        if cand:
            return cand, "tag_answer"

    # 2 Final Answer / Final Asnwer / Final answer などのゆるい抽出
    #    "Final Asnwer" のtypoも拾うため answ\w* にしている
    m = re.search(
        r"(final\s*answ\w*)\s*[:：\-]*\s*([A-Za-z])\b",
        t,
        flags=re.IGNORECASE,
    )
    if m:
        cand = normalize_ans(m.group(2))
        return cand, "final_answer"

    # 3 末尾近辺の単独ラベルを拾う（A〜Zの1文字）
    #    ただし valid_labels があるならそれに含まれるものを優先
    tail = t[-4000:]  # 長文でも末尾中心に見る
    letters = re.findall(r"\b([A-Za-z])\b", tail)
    if letters:
        # 末尾から見て最初に valid_labels に合うものを採用
        for ch in reversed(letters):
            cand = normalize_ans(ch)
            if valid_labels is None or cand in valid_labels:
                return cand, "tail_single_letter"

        # valid_labels が無くて/合わなくて、とりあえず最後の単独文字を返す
        cand = normalize_ans(letters[-1])
        return cand, "tail_single_letter_fallback"

    # 4) 最後の保険：行末に "A" "B"だけの行があるケース
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    for ln in reversed(lines[-50:]):
        m2 = re.fullmatch(r"[\(\[\{<\s]*([A-Za-z])[\)\]\}>\s]*", ln)
        if m2:
            cand = normalize_ans(m2.group(1))
            if valid_labels is None or cand in valid_labels:
                return cand, "line_only_letter"

    return None, "not_found"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="muirbench.json", help="muirbench.json のパス")
    ap.add_argument("--show_fail_examples", type=int, default=0, help="抽出ミス例をN件表示")
    ap.add_argument("--show_wrong_examples", type=int, default=0, help="不正解例をN件表示")
    args = ap.parse_args()

    with open(args.path, "r", encoding="utf-8") as f:
        root = json.load(f)

    gens = list(iter_generations(root))
    if not gens:
        print("ERROR: generations が見つかりませんでした")
        return

    # データに出てくる label を正とする（A-E 以外でも対応）
    observed_labels = set()
    for g in gens:
        lab = g.get("label")
        if isinstance(lab, str) and lab.strip():
            observed_labels.add(lab.strip().upper())

    total = 0
    extracted = 0
    correct = 0

    extraction_fail = 0          # 抽出できなかった
    extraction_invalid = 0       # 抽出できたが、label集合に無い（不正形式扱い）
    wrong = 0

    fail_examples: List[Tuple[str, str]] = []
    wrong_examples: List[Tuple[str, str, str, str]] = []

    rule_counts = {}

    for g in gens:
        total += 1
        label = g.get("label")
        text = g.get("text", "")

        gold = normalize_ans(label) if isinstance(label, str) else ""
        pred, rule = extract_answer(text, valid_labels=observed_labels if observed_labels else None)
        rule_counts[rule] = rule_counts.get(rule, 0) + 1

        if pred is None:
            extraction_fail += 1
            if args.show_fail_examples and len(fail_examples) < args.show_fail_examples:
                snippet = (str(text)[:300] + "…") if len(str(text)) > 300 else str(text)
                fail_examples.append((gold, snippet))
            continue

        pred = normalize_ans(pred)

        if observed_labels and pred not in observed_labels:
            extraction_invalid += 1
            if args.show_fail_examples and len(fail_examples) < args.show_fail_examples:
                snippet = (str(text)[:300] + "…") if len(str(text)) > 300 else str(text)
                fail_examples.append((gold, f"[pred={pred}] {snippet}"))
            continue

        extracted += 1
        if pred == gold:
            correct += 1
        else:
            wrong += 1
            if args.show_wrong_examples and len(wrong_examples) < args.show_wrong_examples:
                snippet = (str(text)[:300] + "…") if len(str(text)) > 300 else str(text)
                wrong_examples.append((gold, pred, rule, snippet))

    acc_extracted = (correct / extracted) if extracted else 0.0
    acc_overall = (correct / total) if total else 0.0

    extraction_miss = extraction_fail + extraction_invalid

    print("=== Result ===")
    print(f"Total generations        : {total}")
    print(f"Extracted (valid)        : {extracted}")
    print(f"Correct                  : {correct}")
    print(f"Wrong                    : {wrong}")
    print(f"Extraction miss (total)  : {extraction_miss}")
    print(f"  - not found            : {extraction_fail}")
    print(f"  - invalid label format : {extraction_invalid}")
    print("")
    print(f"Accuracy (on extracted)  : {acc_extracted:.4f}  ({correct}/{extracted})")
    print(f"Accuracy (overall)       : {acc_overall:.4f}  ({correct}/{total})")

    print("\n=== Extraction rule counts ===")
    for k, v in sorted(rule_counts.items(), key=lambda x: (-x[1], x[0])):
        print(f"{k:28s}: {v}")

    if fail_examples:
        print("\n=== Fail examples (gold, snippet) ===")
        for i, (gold, snip) in enumerate(fail_examples, 1):
            print(f"\n[{i}] gold={gold}\n{snip}")

    if wrong_examples:
        print("\n=== Wrong examples (gold, pred, rule, snippet) ===")
        for i, (gold, pred, rule, snip) in enumerate(wrong_examples, 1):
            print(f"\n[{i}] gold={gold} pred={pred} rule={rule}\n{snip}")


if __name__ == "__main__":
    main()