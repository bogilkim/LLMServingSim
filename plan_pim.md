# Decode PIM GEMV 확장 계획

## 목표와 범위

현재 PIM 경로는 decode attention core만 PIM으로 보내고, NPU에는 prefill
attention만 남긴다. 이 계획의 목표는 **decode 단계의 projection 및 dense
FFN도 선택적으로 PIM에서 실행**할 수 있게 하는 것이다.

첫 번째 구현 대상은 dense decoder-only 모델의 decode-only 실행이며, prefill
GEMM과 MoE expert는 뒤 단계로 분리한다. PIM이 GEMM을 지원하지 않는다는 현재
converter의 전제를 유지하므로, `prefill_chunk > 0`인 토큰은 계속 NPU에서
실행한다.

지원할 연산 단위는 개별 canonical layer가 아니라 데이터 이동을 최소화하는
decode *region*으로 정의한다.

| Region | 포함 레이어 | 이유 |
| --- | --- | --- |
| `attention` | attention core | 현행 동작, 기준선 |
| `qkv` | `qkv_proj` | Q/K/V를 한 번에 생산하는 fused projection |
| `o_proj` | `o_proj` (+ TP ALLREDUCE) | PIM GEMV 뒤 collective 순서를 보존해야 함 |
| `ffn` | `gate_up_proj` → `act_fn` → `down_proj` (+ TP ALLREDUCE) | 중간 activation을 PIM↔NPU로 왕복시키지 않기 위해 하나의 region으로 처리 |

`rotary_emb`, layernorm, sampler는 초기 범위에서는 NPU에 둔다. QKV를 PIM에
두려면 Q/K/V의 PIM→NPU(또는 PIM→PIM attention) 데이터 의존성을 명시적으로
모델링해야 하므로, QKV는 FFN 검증 뒤에 추가한다.

## 성공 기준

1. 기본값과 기존 `--enable-attn-offloading` 실행 결과는 변하지 않는다.
2. decode-only trace에서 활성화한 region은 NPU `COMP_NODE`가 아니라 PIM
   compute node로 표현되고, 그 앞뒤 의존성과 TP collective 순서가 정확하다.
3. PIM weight/activation/KV 용량과 전송량이 같은 메모리 배치 가정에서 일관되게
   계산된다.
4. latency와 에너지 모델은 측정 또는 명시적 보정 파라미터에 근거하며, 미보정
   하드웨어/shape에는 조용히 추정하지 않고 경고 또는 오류를 낸다.
5. 작은 serving 시나리오와 PIM 비활성 회귀 시나리오가 자동 검증된다.

## 0단계 — 모델/하드웨어 계약 결정 및 기준선 확보

이 단계는 코드 변경 전에 완료한다.

1. 대상 PIM의 지원 정밀도, bank/channel 병렬성, GEMV 누산 방식, channel 간
   reduction, activation(GELU/SiLU) 지원 여부를 문서화한다.
2. PIM DRAM에 둘 weight의 영속성 정책을 정한다. 권장 초기 정책은 **모델 시작 시
   고정 weight를 PIM에 resident로 배치**하고, token마다 weight를 다시 전송하지
   않는 것이다. 초기 적재 시간은 steady-state latency와 별도로 보고한다.
3. 모델별 `qkv_proj`, `o_proj`, `gate_up_proj`, `down_proj`의 TP shard shape와
   weight byte 수를 표로 만든다. `head_dim`을 명시적으로 사용하고
   `hidden_size == num_heads * head_dim`을 가정하지 않는다.
4. 각 PIM spec/dtype/shape/channel split에 대해 decode batch size 1 및 대표적인
   연속 배치 크기의 GEMV microbenchmark를 수집한다. FFN은
   `gate_up + activation + down` fused case와 개별 GEMV case를 모두 측정한다.
5. 현 attention-only, NPU-only의 request latency, throughput, energy를 고정
   workload에서 보관하여 이후 비교 기준으로 삼는다.

산출물: PIM 연산 계약서, calibration CSV 또는 config schema, 기준 결과 표.

## 1단계 — 설정과 PIM latency 모델을 일반화

### 1.1 사용자 설정

`--enable-attn-offloading`은 호환성을 위해 유지한다. 새로 다음과 같은
독립적인 선택 인터페이스를 추가한다.

```text
--pim-offload-regions attention,ffn
```

- 기본값은 빈 목록이며, 기존 `--enable-attn-offloading`은 `attention`을 추가하는
  별칭으로 해석한다.
- cluster instance에도 `pim_offload_regions`를 둘 수 있게 하고 CLI와의 우선순위를
  명확히 한다.
- 허용값과 제약을 시작 시 검증한다. 예: `qkv`는 attention과 함께만 허용하고,
  prefill 토큰에는 적용하지 않는다.
- 실행 로그/결과 metadata에 최종 region 목록, latency calibration 버전, PIM
  weight residency를 기록한다.

### 1.2 `PIMModel` API

`serving/core/pim_model.py`의 attention 전용 `get_pim_latency(...)`를 유지하되,
다음처럼 operation별 API로 확장한다.

```python
get_attention_latency(...)
get_gemv_latency(op, m, n, batch_tokens, dtype, channel_split, tp_size)
get_ffn_latency(hidden_size, intermediate_size, batch_tokens, dtype,
                channel_split, tp_size, fused=True)
get_pim_energy(op, latency_ns, bytes_read, bytes_written)
```

- 숫자를 코드에 하드코딩하지 말고 PIM spec별 calibration table을 읽는다.
- 보간/외삽의 기준과 유효 shape 범위를 metadata로 저장한다.
- `channel_split`은 단순히 latency를 채널 수로 나누지 말고, 사용 가능한
  output/reduction 병렬성과 tail channel을 반영한다.
- latency에는 command/setup, DRAM read, PIM MAC, channel/reduction, output write를
  구분해 기록한다. 이 항목들이 있어야 NPU↔PIM 전송과 중복 계산을 피할 수 있다.

### 1.3 메모리 배치와 용량

`config_builder.py`의 placement schema를 확장해 PIM weight residency를 표현한다.
기존 `weights: cpu`가 "NPU가 CPU memory에서 weight를 읽음"을 의미할 수 있으므로,
PIM 실행 가능 weight라는 의미를 별도 필드로 혼합하지 않는다. 예를 들어
`pim_weights: resident`와 `pim_region`을 layer/block 규칙에 추가한다.

- PIM DRAM의 KV, PIM-resident weight, 작업 activation이 합쳐서 채널별 용량을
  넘지 않는지 시작 시 검증한다.
- TP에서는 각 rank의 weight shard만 각 rank의 PIM channel 집합에 배치한다.
- `memory_model.py`에 PIM-resident weight와 일시 activation accounting을 추가한다.
  request별 KV cache accounting과 모델 lifetime weight accounting을 분리한다.
- 초기 구현에서는 부족한 PIM 용량일 때 자동 eviction하지 말고 명확한 오류를 낸다.

## 2단계 — trace IR과 Chakra converter를 generic PIM region으로 변경

현 converter는 `PIM` marker 뒤 연산을 attention으로 간주하고, weight parent를
명시적으로 버리며, `"attn"` 이름으로 남은 prefill attention을 판별한다.
따라서 trace generator만 바꾸면 GEMV weight 의존성과 연산 순서가 틀어진다.

다음 순서로 converter를 고친다.

1. 텍스트 trace의 `PIM` marker에 region/mode를 전달한다. 기존 `PIM {channel}`은
   계속 읽되, 새 형식은 예를 들어 `PIM {channel} GEMV` 또는 명시적
   `PIM_BEGIN`/`PIM_END`로 versioning한다.
2. marker 내부의 각 layer가 input, output, **weight location/size**, operation
   kind를 가질 수 있게 `Layer`와 PIM node 생성 인터페이스를 확장한다.
3. generic dependency 규칙을 구현한다.
   - 직전 NPU compute/collective 또는 입력 load → 첫 PIM node
   - PIM-resident weight는 초기 적재/상주 node에 의존하고, 비상주 weight는 실제
     memory-load node에 의존
   - PIM node들의 region 내부 순서 보장
   - 마지막 PIM node → 다음 NPU compute/collective
4. `o_proj`와 `down_proj`의 PIM compute 뒤에는 기존과 같은 TP ALLREDUCE를 붙인다.
5. PIM channel parallel 작업의 fan-out/fan-in과 channel reduction을 graph에
   표현한다. 모델이 이를 내부 latency에 포함한다면 explicit reduction node를 만들지
   않는다는 선택을 문서화한다.
6. `attn_remain`, 문자열 `"attn"` 판별, attention weight 예외처럼 attention에만
   묶인 분기를 region metadata 기반으로 제거한다. 기존 attention-only trace에 대한
   converter regression fixture를 반드시 유지한다.

이 변경은 `astra-sim/` integration 변경이므로 별도, 작고 독립적인 commit으로
검증한다.

## 3단계 — trace generator의 decode region emission

`serving/core/trace_generator.py`에 PIM region 선택과 emission을 추가한다.

1. `BatchCtx`에 `decode_tokens`와 PIM channel별 work assignment를 일반화한다.
   현 `decode_lens`는 attention의 sequence-length balancing 전용이므로 GEMV의
   token/output-channel partition을 재사용하지 않는다.
2. mixed batch에서 **decode 부분만** PIM으로 보내도록 layer별 work를 분리한다.
   `total_len` 하나를 공유하면 prefill GEMM이 우연히 PIM으로 갈 수 있으므로,
   `prefill_tokens`와 `decode_tokens`를 명시적으로 전달한다.
3. `_emit_pim_attention`과 별개로 `_emit_pim_gemv_region`을 만든다. NPU emission을
   통째로 0-latency layer로 남기지 말고, 해당 decode work의 NPU layer를 trace에서
   제외한다.
4. 우선 `ffn` region을 구현한다.
   - prefill의 `gate_up_proj → act_fn → down_proj`는 현재 NPU emission 유지
   - decode의 세 연산은 PIM fused FFN node/region으로 생성
   - PIM output이 이후 layernorm 또는 다음 block으로 정확히 연결되는지 검증
   - `down_proj` 뒤 TP ALLREDUCE는 PIM output을 입력으로 사용
5. 다음으로 `o_proj`를 구현하고, 그 뒤 `qkv`를 구현한다. `qkv`는 rotary 및
   PIM attention과의 data location/의존성을 설계한 뒤에만 활성화한다.
6. 기존 power accumulator를 operation별 PIM latency/read/write byte를 받도록
   확장한다. 현 attention PIM 에너지 계산과 합산 규칙은 유지한다.
7. draft/verify, EAGLE3, sub-batch interleaving은 처음에는 새 PIM GEMV region을
   명시적으로 거부한다. 일반 decode 경로 검증 후 지원 여부를 별도 설계한다.

## 4단계 — 검증 계층

### 단위/구조 검증

- PIM offload region 파서와 legacy flag 호환성
- PIM capacity: 정상, 정확히 맞음, 부족함, TP shard별 경우
- latency table lookup: 유효 범위, 보간, 미지원 spec/shape 오류
- FFN/QKV shape 및 byte 계산: GQA, 명시적 `head_dim`, TP > 1
- trace snapshot: NPU-only, attention-only, `ffn`, `attention,ffn`, mixed batch
- Chakra graph inspection: PIM weight dependency, PIM→ALLREDUCE, PIM→다음 NPU
  dependency, prefill이 PIM으로 가지 않음을 검증

### 시뮬레이터 통합 검증

1. 단일 NPU, `--skip-prefill`, 단일 request로 FFN-only PIM trace/graph를 확인한다.
2. mixed prefill+decode batch에서 prefill FFN은 NPU, decode FFN만 PIM인지 확인한다.
3. TP=2에서 `o_proj`/`down_proj` PIM compute 뒤 ALLREDUCE의 크기와 순서를 확인한다.
4. PIM capacity 부족 config가 실행 초기에 설명 가능한 오류로 중단되는지 확인한다.
5. PIM 비활성, attention-only의 결과가 변경 전 baseline과 일치하는지 회귀한다.

### 정확도/성능 검증

- microbenchmark latency와 simulator의 PIM GEMV estimate를 shape별로 비교한다.
  p50/p90 오차 목표를 사전에 정하고 결과 metadata에 남긴다.
- 동일 workload에서 NPU-only / attention-only / FFN-only / attention+FFN의 TTFT,
  TPOT, throughput, PIM/NPU energy를 비교한다.
- PIM↔NPU activation 전송, weight initial load, steady-state execution을 별도
  지표로 보고하여 과도한 speedup을 방지한다.

## 5단계 — 확대와 문서화

1. FFN과 dense projection 검증이 끝난 뒤 QKV→rotary→attention→O-proj를 한 PIM
   pipeline으로 최적화할지 결정한다.
2. MoE expert GEMV는 별도 설계로 다룬다. EP all-to-all, expert residency,
   load imbalance 때문에 dense FFN path와 섞지 않는다.
3. sub-batch interleaving은 PIM region별 overlap 가능 여부를 정의한 뒤 지원한다.
4. `docs/`에 region 설정, 하드웨어 가정, calibration 방법, 용량 제약, 결과 해석을
   추가한다. README에는 상세 옵션을 추가하지 않는다.

## 권장 커밋 순서

1. `Add configurable PIM offload regions and calibration schema`
2. `Model PIM-resident weights and capacity`
3. `Generalize Chakra PIM regions and dependencies`
4. `Offload decode dense FFN to PIM`
5. `Add PIM output projection and TP collective coverage`
6. `Add decode QKV PIM pipeline`
7. `Document and validate PIM GEMV offloading`

각 커밋에는 해당 단계의 trace fixture 또는 작은 `python -m serving` 검증 명령과
출력 경로를 함께 남긴다.

## 먼저 결정이 필요한 사항

구현 시작 전 아래 세 항목은 확정해야 한다.

1. 사용할 PIM hardware/model에서 GEMV, reduction, SiLU/GELU를 실제로 지원하는가?
   지원하지 않으면 FFN은 GEMV 두 번과 NPU activation으로 쪼개야 하며, 그 경우
   activation 왕복 비용을 반드시 포함해야 한다.
2. PIM DRAM 용량 중 model weight에 예약할 양과 KV cache에 예약할 양은 얼마인가?
3. 결과가 steady-state decode latency만 필요한가, 아니면 PIM weight initial load와
   cold-start latency도 포함해야 하는가?
