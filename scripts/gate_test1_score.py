import argparse, csv, json
from pathlib import Path
from loguru import logger

def load_blind(p):
    labels = {}
    with open(p, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            w = row["window_id"].strip()
            l = row["your_label(H/N)"].strip().upper()
            if l in ("H","N"): labels[w] = l
    return labels

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="data/gate_test1")
    args = p.parse_args()
    d = Path(args.dir)
    blind = load_blind(d / "label_blind.csv")
    answers = json.loads((d / "answers.json").read_text(encoding="utf-8"))
    tp=fp=fn=tn=0
    for wid in sorted(blind):
        if wid not in answers: continue
        h, t = blind[wid], answers[wid]["true_label"]
        e = answers[wid]["engagement"]
        hit = "OK" if h==t else "MISS"
        logger.info(f"  {wid} | human:{h} true:{t} eng:{e:.2f} {hit}")
        if h=="H" and t=="H": tp+=1
        elif h=="H" and t=="N": fp+=1
        elif h=="N" and t=="H": fn+=1
        else: tn+=1
    total=tp+fp+fn+tn
    acc=(tp+tn)/max(total,1)
    prec=tp/max(tp+fp,1)
    rec=tp/max(tp+fn,1)
    logger.info(f"accuracy: {acc:.1%} ({tp+tn}/{total})")
    logger.info(f"precision(H): {prec:.1%} | recall(H): {rec:.1%}")
    logger.info(f"confusion: TP={tp} FP={fp} FN={fn} TN={tn}")
    if acc<=0.55: logger.info("RED: no signal in 5 frames")
    elif acc<=0.70: logger.info("YELLOW: weak signal")
    else: logger.info("GREEN: signal sufficient")

if __name__=="__main__": main()
