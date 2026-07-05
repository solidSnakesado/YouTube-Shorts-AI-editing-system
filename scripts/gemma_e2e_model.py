# 48일차: gemma_e2e_model.py — 수정 7회
#   수정 1회: L45~57 forward 화이트리스트를 Colab 스모크 실키로 교체
#   수정 2회(OOM 대응): logits_to_keep=1 추가, use_gradient_checkpointing 추가
#   수정 3회(dtype 충돌): forward에서 float 입력 bf16 캐스팅
#   수정 4회(eval dtype 충돌): forward를 bf16 autocast로 감싸 학습/평가 경로 통일
#   수정 5회(50일차, 로컬 로드 실패): load_model_for_infer에 device_map={"": 0} 추가
#     (12GB 추정 초과로 인한 CPU 오프로드 시도 차단, bnb 4bit ValueError 해소)
#   수정 6회(50일차, 스래싱 완화): 최종 norm hook으로 마지막 층만 캡처 (수정본 기준 L70~77, L89~104)
#   수정 7회(50일차, PLE CPU 상주): 수정본 기준 L152~196 래퍼+헬퍼 신설, L199~200·L213~218 적용
# 레포 경로: yt_shorts_ai/scripts/gemma_e2e_model.py
# 역할: 방안 1(round12) 모델 래퍼
#   - Gemma 4 E4B 4bit 로드 (QLoRA: 학습·추론 양자화 상태 일치, 로컬 12GB 추론 대비)
#   - 언어/융합층 LoRA 부착 (비전·오디오 타워 동결)
#   - 마지막 hidden state → 마스크 평균 풀링 → 회귀 헤드(MLP) → hook_score 실수 출력
#   - lm_head / 텍스트 생성 미사용 (층1·2·4 원천 제거)
#   - 저장물: LoRA 어댑터 + 헤드 state_dict + 정규화 통계(mu/sd) — round12 격리 경로

import json
import os

import torch
import torch.nn as nn

MODEL_NAME = "unsloth/gemma-4-E4B-it"
HIDDEN_SIZE = 2560          # 48일차: Gemma 4 E4B 언어모델 hidden 차원 (C-1 vis 임베딩과 동일)
HEAD_MID = 512              # 48일차: 회귀 헤드 중간 차원
HEAD_DROPOUT = 0.1


# ---------------------------------------------------------------------------
# 회귀 헤드: LayerNorm → Linear → GELU → Dropout → Linear(→1)
# fp32 고정 (4bit/bf16 본체와 분리, 수치 안정)
# ---------------------------------------------------------------------------
class RegressionHead(nn.Module):
    def __init__(self, hidden_size: int = HIDDEN_SIZE):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, HEAD_MID)
        self.act = nn.GELU()
        self.drop = nn.Dropout(HEAD_DROPOUT)
        self.fc2 = nn.Linear(HEAD_MID, 1)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        # pooled: [B, hidden] fp32
        x = self.norm(pooled)
        x = self.drop(self.act(self.fc1(x)))
        return self.fc2(x).squeeze(-1)  # [B]


# ---------------------------------------------------------------------------
# E2E 래퍼: 멀티모달 입력 → 점수
# ---------------------------------------------------------------------------
class GemmaE2EScorer(nn.Module):
    # 48일차: batch에서 본체 forward로 넘길 키 화이트리스트
    #   (수정 1회: Colab CPU 스모크로 Gemma4Processor 실키 확정 반영 —
    #    image_position_ids / mm_token_type_ids 추가, token_type_ids 제거)
    _MODEL_KEYS = (
        "input_ids",
        "attention_mask",
        "pixel_values",
        "image_position_ids",
        "input_features",
        "input_features_mask",
        "mm_token_type_ids",
    )

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base = base_model
        self.head = RegressionHead().float()
        # 50일차 수정 6회: 최종 norm 모듈 탐색 — hook으로 마지막 층만 캡처해
        #   output_hidden_states=True의 전 층(~37개, ~0.6GB) 상주 보관을 제거.
        #   HF Gemma 구현상 hidden_states[-1] == 최종 norm 출력 → 수치 동일(정합 무해).
        self._final_norm = None
        for name, mod in base_model.named_modules():
            if name.endswith("language_model.norm"):
                self._final_norm = mod
                break

    def forward(self, batch: dict) -> torch.Tensor:
        # 48일차 수정 3회: 프로세서 fp32 출력(pixel_values/input_features)을 bf16으로
        #   캐스팅 — batch>1 패딩 경로에서 layer_norm dtype 충돌 방지
        inputs = {
            k: (v.to(torch.bfloat16) if v.is_floating_point() else v)
            for k, v in batch.items() if k in self._MODEL_KEYS
        }
        # 48일차 수정 4회: bf16 autocast로 학습/평가 경로 dtype 통일
        #   (unsloth가 fp32로 유지하는 norm 가중치 ↔ bf16 입력 충돌이 eval 경로에서 발생)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # 50일차 수정 6회: norm hook 경로 (미탐색 시 기존 전 층 보관 경로 폴백)
            captured: dict = {}
            handle = None
            if self._final_norm is not None:
                handle = self._final_norm.register_forward_hook(
                    lambda _m, _i, o: captured.__setitem__("h", o))
            out = self.base(
                **inputs,
                output_hidden_states=(self._final_norm is None),
                use_cache=False,
                logits_to_keep=1,   # 48일차 수정 2회: lm_head 로짓을 1위치만 계산 (전량 계산 시 OOM)
            )
            if handle is not None:
                handle.remove()
            hidden = captured.get("h") if self._final_norm is not None \
                else out.hidden_states[-1]                # [B, T, H] (bf16)
            mask = batch["attention_mask"].unsqueeze(-1)  # [B, T, 1]
            mask = mask.to(hidden.dtype)
            # 48일차: 패딩 제외 평균 풀링
            summed = (hidden * mask).sum(dim=1)           # [B, H]
            count = mask.sum(dim=1).clamp(min=1.0)        # [B, 1]
        pooled = (summed / count).float()                 # autocast 밖 fp32로 헤드 통과
        return self.head(pooled)                          # [B]


# ---------------------------------------------------------------------------
# 로드: 4bit base + LoRA (학습용) / 추론용은 어댑터 경로 지정
# ---------------------------------------------------------------------------
def load_model_for_train(
    max_seq_length: int = 8192,
    lora_r: int = 16,
    lora_alpha: int = 16,
):
    # 48일차: Unsloth 4bit 로드 — Colab 기본 torch 보존 전제 (설치 순서 핸드오프 §5 참조)
    from unsloth import FastModel

    base, processor = FastModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )
    # 48일차: 비전/오디오 타워 동결, 언어·융합층만 LoRA
    base = FastModel.get_peft_model(
        base,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",  # 48일차 수정 2회: 활성화 (OOM 대응)
        random_state=42,
    )
    model = GemmaE2EScorer(base)
    # 48일차: 헤드는 전체 학습 (fp32), base 쪽은 LoRA 파라미터만 requires_grad
    for p in model.head.parameters():
        p.requires_grad = True
    return model, processor


# ---------------------------------------------------------------------------
# 50일차 수정 7회: PLE(층별 임베딩) CPU 상주 래퍼 — 로컬 12GB 스래싱 해소
#   근거: base 발자국 분해 실측 — embed_tokens_per_layer가 5.64GB(BF16)로 전체의 절반.
#   PLE는 행렬곱이 아닌 토큰별 조회(lookup)라 CPU 상주 + 결과만 GPU 전송이 저비용
#   (~55MB/샘플). 가중치·조회값 동일 → 정합 무해. 추론 경로 전용(학습 경로 무수정).
# ---------------------------------------------------------------------------
class _CPUOffloadModule(nn.Module):
    """내부 모듈을 CPU에 상주시키고, 텐서 입력을 CPU로/출력을 원 디바이스로 이동."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner.to("cpu")

    def forward(self, *args, **kwargs):
        dev = None
        def _mv(x):
            nonlocal dev
            if torch.is_tensor(x):
                if dev is None and x.is_cuda:
                    dev = x.device
                return x.to("cpu")
            return x
        args = [_mv(a) for a in args]
        kwargs = {k: _mv(v) for k, v in kwargs.items()}
        out = self.inner(*args, **kwargs)
        if dev is not None and torch.is_tensor(out):
            out = out.to(dev)
        return out


def _offload_ple_to_cpu(base: nn.Module) -> bool:
    """embed_tokens_per_layer 모듈을 찾아 CPU 상주 래퍼로 교체. 성공 시 True."""
    target_name = None
    for name, _mod in base.named_modules():
        if name.endswith("embed_tokens_per_layer"):
            target_name = name
            break
    if target_name is None:
        return False
    parent = base
    parts = target_name.split(".")
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], _CPUOffloadModule(getattr(parent, parts[-1])))
    return True


def load_model_for_infer(adapter_dir: str, max_seq_length: int = 8192,
                         ple_offload: bool = True):
    # 48일차: 로컬 12GB 추론용 — 4bit base + 저장된 LoRA 어댑터 + 헤드
    # 50일차 수정 5회: device_map={"": 0} 강제 — transformers의 보수적 메모리 추정이
    #   11.94GB에서 CPU 오프로드를 시도 → bnb 4bit 거부(ValueError)로 로드 실패.
    #   실측 4bit 가중치는 ~8GB이므로 GPU 단일 배치를 강제해 관문 실측.
    from unsloth import FastModel

    base, processor = FastModel.from_pretrained(
        model_name=adapter_dir,          # 어댑터 경로 지정 시 base+adapter 함께 로드
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        device_map={"": 0},              # 50일차: 전 모듈 GPU 0 강제 (오프로드 차단)
    )
    # 50일차 수정 7회: PLE만 CPU 상주 (GPU 상주 11.12GB → 약 5.5GB 목표)
    if ple_offload:
        moved = _offload_ple_to_cpu(base)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"PLE CPU 상주: {'적용' if moved else '모듈 미발견 — 미적용(기존 경로)'}")
    model = GemmaE2EScorer(base)
    head_path = os.path.join(adapter_dir, "regression_head.pt")
    state = torch.load(head_path, map_location="cpu")
    model.head.load_state_dict(state)
    model.head.float()
    model.eval()
    return model, processor


# ---------------------------------------------------------------------------
# 저장/로드: 어댑터 + 헤드 + 정규화 통계 (round12 격리 경로에 한 세트로)
# ---------------------------------------------------------------------------
def save_checkpoint(model: GemmaE2EScorer, out_dir: str, norm_stats: dict):
    # norm_stats: {"mu": float, "sd": float} — 타깃 표준화 통계 (추론 정합 필수)
    os.makedirs(out_dir, exist_ok=True)
    model.base.save_pretrained(out_dir)  # LoRA 어댑터만 저장 (PEFT)
    torch.save(model.head.state_dict(), os.path.join(out_dir, "regression_head.pt"))
    with open(os.path.join(out_dir, "norm_stats.json"), "w", encoding="utf-8") as f:
        json.dump(norm_stats, f, ensure_ascii=False, indent=2)


def load_norm_stats(adapter_dir: str) -> dict:
    with open(os.path.join(adapter_dir, "norm_stats.json"), encoding="utf-8") as f:
        return json.load(f)


def count_trainable(model: nn.Module) -> str:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"trainable {trainable:,} / total {total:,} ({100 * trainable / total:.3f}%)"