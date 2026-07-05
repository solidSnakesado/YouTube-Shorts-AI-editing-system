import json, os, subprocess, time, sys
D = sys.argv[1] if len(sys.argv) > 1 else "datasets/gemma_audio_v2/dataset_neg.jsonl"
PAT = sys.argv[2] if len(sys.argv) > 2 else "run_gemma_neg.py"
T = 1959
try:
    out = subprocess.check_output(["pgrep", "-f", PAT]).decode().split()
    P = int(out[-1]) if out else None
except Exception:
    P = None
v, s = set(), 0
if os.path.exists(D):
    for l in open(D):
        if l.strip():
            s += 1
            try: v.add(json.loads(l)["metadata"]["video_id"])
            except: pass
d = len(v)
def et(p):
    if not p: return 0
    try: return int(subprocess.check_output(["ps","-o","etimes=","-p",str(p)]).decode().strip() or 0)
    except: return 0
e = et(P)
print("빌드 상태 :", ("PID %d 실행중" % P) if P else "멈춤/미시작")
print("처리 영상 : %d/%d (%.1f%%)" % (d, T, d/T*100 if T else 0))
print("누적 샘플 : %d개%s" % (s, (" (영상당 %.2f)" % (s/d)) if d else ""))
if e and d:
    eta = (T-d)*(e/d)
    print("경과/속도 : %dh%dm (영상당 %.1fs)" % (e//3600, e%3600//60, e/d))
    print("잔여/완료 : %.1fh -> %s" % (eta/3600, time.strftime("%H:%M", time.localtime(time.time()+eta))))
    print("최종 예상 : 약 %d개" % int(s/d*T))
