/**
 * Synapse SNN-RAG 하이브리드 엔진 (Spiking Neural Network + Neural RAG Hybrid)
 *
 * 아키텍처 파이프라인:
 * 1단계 (Event-Driven SNN Gate):
 *   - 문서를 LIF(Leaky Integrate-and-Fire) 뉴런으로 모델링
 *   - 쿼리 토큰/N-gram을 스파이크 트레인(Rate + Latency Coding)으로 인코딩
 *   - BM25/IDF 기반 시냅스 + STDP(Spike-Timing-Dependent Plasticity) 학습 가중치로
 *     관련 뉴런에만 선택적으로 전류 주입
 *   - 측면 억제(Lateral Inhibition) 및 다중 홉 시냅스 전파(Multi-Hop Spreading Activation) 수행
 *   - 발화(Spike)한 극소수 활성 후보 뉴런만 2단계로 전달 → 신경망 연산량 대폭 절감
 *
 * 2단계 (Neural Dense & Reranker Hybrid Fusion):
 *   - 1단계에서 발화한 소수 후보에만 플러그형 신경망 임베더(Dense)와 리랭커(Cross-Encoder) 적용
 *   - 외부 신경망 장애 시 try-catch로 로컬 Int8 폴백 무중단 전환
 *   - SNN 발화 점수 + BM25 + Dense 유사도 + Reranker 점수를 가중 융합
 *   - 선택적 Active Inference 토큰 예산 최적화로 최종 컨텍스트 선택
 */

import { localHashEmbed, type VectorEmbedderFn } from './ai-vector';
import { ActiveInferenceContextSelector, type ContextCandidate } from './causal-concepts';

export const SNN_RAG_CONFIG = {
  /** LIF 시뮬레이션 타임스텝 수 */
  DEFAULT_TIME_STEPS: 6,
  /** 막전위 누설 계수 (0 < lambda < 1) */
  DEFAULT_LEAK_FACTOR: 0.78,
  /** 뉴런 발화 임계 전위 */
  DEFAULT_FIRING_THRESHOLD: 0.55,
  /** 발화 후 불응기 스텝 수 */
  DEFAULT_REFRACTORY_STEPS: 1,
  /** 측면 억제 강도 (승자 뉴런이 경쟁 뉴런을 억제하는 비율) */
  DEFAULT_LATERAL_INHIBITION: 0.12,
  /** 다중 홉 시냅스 전파 감쇠 계수 */
  DEFAULT_HOP_DECAY: 0.45,
  /** 최대 시냅스 전파 홉 수 */
  DEFAULT_MAX_HOPS: 2,
  /** STDP 장기 강화(LTP) 학습률 */
  DEFAULT_STDP_LTP: 0.28,
  /** STDP 장기 약화(LTD) 학습률 */
  DEFAULT_STDP_LTD: 0.15,
  /** 시냅스 가중치 하한/상한 */
  DEFAULT_MIN_SYNAPSE_WEIGHT: 0.1,
  DEFAULT_MAX_SYNAPSE_WEIGHT: 4.0,
  /** BM25 하이퍼파라미터 */
  DEFAULT_BM25_K1: 1.5,
  DEFAULT_BM25_B: 0.75,
  /** 기본 반환 Top-K */
  DEFAULT_TOP_K: 5,
  /** 1단계 SNN 게이트가 2단계 신경망으로 넘길 후보 배수 (topK * multiplier) */
  DEFAULT_STAGE1_MULTIPLIER: 4,
  /** 최소 1단계 후보 수 */
  DEFAULT_MIN_STAGE1_CANDIDATES: 8,
  /** 수상돌기 Dense 기저 전류 상위 수용 뉴런 수 */
  DEFAULT_DENDRITIC_TOP_K: 24,
  /** 수상돌기 Dense 기저 전류 이득 계수 */
  DEFAULT_DENDRITIC_GAIN: 0.55,
  /** 기본 점수 융합 가중치 */
  WEIGHTS: {
    SNN: 0.35,
    BM25: 0.15,
    DENSE: 0.25,
    RERANK: 0.25,
  },
  /** 토큰 비용 추정 계수 (문자 수 대비 대략적 토큰 환산) */
  CHARS_PER_TOKEN_ESTIMATE: 3.5,
  /** 기본 컨텍스트 추정 최소 토큰 */
  MIN_TOKEN_COST: 8,
} as const;

export interface SnnRagDocument {
  readonly id: string;
  readonly text: string;
  readonly title?: string;
  /** 다른 문서 ID로의 시냅스(인과/참조) 연결 목록 */
  readonly links?: readonly string[];
  /** 태그 또는 핵심 개념 키워드 */
  readonly tags?: readonly string[];
  /** 사전 계산된 토큰 비용 (미지정 시 텍스트 길이 기반 자동 추정) */
  readonly tokenCost?: number;
  readonly metadata?: Record<string, unknown>;
}

export interface NeuralRerankCandidate {
  readonly id: string;
  readonly text: string;
  readonly title?: string;
  readonly snnScore: number;
  readonly denseScore: number;
}

/**
 * 플러그형 신경망 리랭커 인터페이스 (Cross-Encoder, BGE-Reranker, LLM Reranker 등)
 * 반환값: 문서 ID -> 0..1 정규화 관련성 점수 맵
 */
export type NeuralRerankerFn = (
  query: string,
  candidates: readonly NeuralRerankCandidate[]
) => Record<string, number> | Map<string, number> | Promise<Record<string, number> | Map<string, number>>;

export interface SnnRagOptions {
  readonly timeSteps?: number;
  readonly leakFactor?: number;
  readonly firingThreshold?: number;
  readonly refractorySteps?: number;
  readonly lateralInhibition?: number;
  readonly hopDecay?: number;
  readonly maxHops?: number;
  readonly stdpLtpRate?: number;
  readonly stdpLtdRate?: number;
  /** 플러그형 신경망 문서(Passage) Dense 임베더 (미지정 시 로컬 Int8 해시 임베더 사용) */
  readonly neuralEmbedder?: VectorEmbedderFn;
  /** 비대칭 모델(E5/BGE 등)용 쿼리 전용 임베더 (미지정 시 neuralEmbedder 공유) */
  readonly queryEmbedder?: VectorEmbedderFn;
  /** 플러그형 신경망 리랭커 (미지정 시 SNN + Dense + BM25 융합 사용) */
  readonly neuralReranker?: NeuralRerankerFn;
  /** 점수 융합 가중치 재정의 */
  readonly weights?: Partial<typeof SNN_RAG_CONFIG.WEIGHTS>;
}

export interface SnnRagQueryOptions {
  readonly topK?: number;
  /** 1단계 SNN 게이트에서 2단계 신경망으로 통과시킬 최대 후보 수 */
  readonly maxNeuralCandidates?: number;
  /** 토큰 예산을 지정하면 Active Inference 선택기로 예산 내 최적 조합 반환 */
  readonly tokenBudget?: number;
  /** 2단계 신경망 임베딩/리랭커 비활성화 여부 (순수 SNN 모드) */
  readonly pureSnnOnly?: boolean;
  /** 사전 임베딩된 벡터가 있을 때 수상돌기 기저 전류 주입 활성화 여부 (기본 true) */
  readonly enableDendriticCurrent?: boolean;
}

export interface SnnRagHit {
  readonly id: string;
  readonly score: number;
  readonly snnScore: number;
  readonly bm25Score: number;
  readonly denseScore: number;
  readonly rerankScore: number;
  readonly spikeCount: number;
  readonly firstSpikeStep: number;
  readonly hopDistance: number;
  readonly tokenCost: number;
  readonly title?: string;
  readonly text: string;
  readonly metadata?: Record<string, unknown>;
}

export interface SnnRagTelemetry {
  readonly totalDocuments: number;
  readonly activeNeuronsEvaluated: number;
  readonly firedNeurons: number;
  /** 발화하지 않은 뉴런 비율 (0..1, 높을수록 스파이크 희소성 우수) */
  readonly spikeSparsity: number;
  readonly neuralCandidatesEvaluated: number;
  /** 전체 문서 대비 2단계 신경망 연산 절감 비율 (0..1) */
  readonly neuralComputeSavedRatio: number;
  readonly stage1SnnMs: number;
  readonly stage2NeuralMs: number;
  readonly totalMs: number;
  readonly fallbackTriggered: boolean;
  readonly warningMessage?: string;
}

export interface SnnRagSearchResult {
  readonly hits: readonly SnnRagHit[];
  readonly selectedByBudgetIds?: readonly string[];
  readonly totalSelectedTokens?: number;
  readonly telemetry: SnnRagTelemetry;
}

export interface StdpFeedbackResult {
  readonly updatedSynapses: number;
  readonly strengthenedDocs: number;
  readonly weakenedDocs: number;
}

interface IndexedNeuronDoc {
  readonly doc: SnnRagDocument;
  readonly tokens: readonly string[];
  readonly termFreqs: Map<string, number>;
  readonly docLength: number;
  readonly tokenCost: number;
  cachedDenseVector?: Float32Array;
  cachedDenseNorm?: number;
}

interface SnnActivationState {
  readonly docId: string;
  readonly spikeCount: number;
  readonly firstSpikeStep: number;
  readonly peakPotential: number;
  readonly hopDistance: number;
  readonly snnScore: number;
  readonly bm25Score: number;
}

/**
 * 한국어/영문/코드 식별자 복합 토큰화 + 부분 문자 N-gram 생성
 */
export function tokenizeForSnn(text: string): string[] {
  const normalized = text.toLowerCase().trim();
  if (!normalized) return [];

  const rawTokens = normalized.split(/[^a-z0-9가-힣_]+/).filter((t) => t.length > 0);
  const out: string[] = [];

  for (const tok of rawTokens) {
    out.push(tok);
    // 한글 복합어/활용형 매칭 강화를 위해 2글자 이상 한글 토큰에 바이그램 추가
    if (/[가-힣]/.test(tok) && tok.length >= 3) {
      for (let i = 0; i <= tok.length - 2; i++) {
        out.push(`~${tok.slice(i, i + 2)}`);
      }
    }
  }
  return out;
}

function toFloat32AndNorm(vec: readonly number[]): { arr: Float32Array; norm: number } {
  const arr = new Float32Array(vec.length);
  let sumSq = 0;
  for (let i = 0; i < vec.length; i++) {
    const v = Number.isFinite(vec[i]) ? vec[i] : 0;
    arr[i] = v;
    sumSq += v * v;
  }
  return { arr, norm: Math.sqrt(sumSq) || 1 };
}

function cosineSimilarityFloat32(
  a: Float32Array,
  aNorm: number,
  b: Float32Array,
  bNorm: number
): number {
  const len = Math.min(a.length, b.length);
  if (len === 0) return 0;
  let dot = 0;
  for (let i = 0; i < len; i++) {
    dot += a[i] * b[i];
  }
  const sim = dot / (aNorm * bNorm);
  // -1..1 범위를 0..1 범위로 정규화
  return Math.max(0, Math.min(1, (sim + 1) * 0.5));
}

function estimateTokenCost(text: string, explicit?: number): number {
  if (typeof explicit === 'number' && Number.isFinite(explicit) && explicit > 0) {
    return Math.round(explicit);
  }
  return Math.max(
    SNN_RAG_CONFIG.MIN_TOKEN_COST,
    Math.ceil(text.length / SNN_RAG_CONFIG.CHARS_PER_TOKEN_ESTIMATE)
  );
}

/**
 * SNN 시냅스 레이어 (LIF 뉴런 시뮬레이션, 측면 억제, 다중 홉 전파, STDP 가소성)
 */
export class SnnSynapticLayer {
  private readonly timeSteps: number;
  private readonly leakFactor: number;
  private readonly firingThreshold: number;
  private readonly refractorySteps: number;
  private readonly lateralInhibition: number;
  private readonly hopDecay: number;
  private readonly maxHops: number;
  private readonly stdpLtpRate: number;
  private readonly stdpLtdRate: number;

  /** 토큰 -> (docId -> 기본 TF) 역색인 */
  private readonly invertedIndex = new Map<string, Map<string, number>>();
  /** STDP 온라인 학습 시냅스 가중치: `${token}\0${docId}` -> weight (기본 1.0) */
  private readonly synapseWeights = new Map<string, number>();
  /** 문서 간 양방향 시냅스 연결 그래프: docId -> Set<neighborDocId> */
  private readonly docGraph = new Map<string, Set<string>>();

  private totalDocLength = 0;
  private docCount = 0;

  constructor(options?: SnnRagOptions) {
    this.timeSteps = Math.max(1, options?.timeSteps ?? SNN_RAG_CONFIG.DEFAULT_TIME_STEPS);
    this.leakFactor = Math.max(0.05, Math.min(0.99, options?.leakFactor ?? SNN_RAG_CONFIG.DEFAULT_LEAK_FACTOR));
    this.firingThreshold = Math.max(0.05, options?.firingThreshold ?? SNN_RAG_CONFIG.DEFAULT_FIRING_THRESHOLD);
    this.refractorySteps = Math.max(0, options?.refractorySteps ?? SNN_RAG_CONFIG.DEFAULT_REFRACTORY_STEPS);
    this.lateralInhibition = Math.max(
      0,
      Math.min(0.8, options?.lateralInhibition ?? SNN_RAG_CONFIG.DEFAULT_LATERAL_INHIBITION)
    );
    this.hopDecay = Math.max(0, Math.min(0.95, options?.hopDecay ?? SNN_RAG_CONFIG.DEFAULT_HOP_DECAY));
    this.maxHops = Math.max(0, Math.min(4, options?.maxHops ?? SNN_RAG_CONFIG.DEFAULT_MAX_HOPS));
    this.stdpLtpRate = Math.max(0.01, options?.stdpLtpRate ?? SNN_RAG_CONFIG.DEFAULT_STDP_LTP);
    this.stdpLtdRate = Math.max(0.01, options?.stdpLtdRate ?? SNN_RAG_CONFIG.DEFAULT_STDP_LTD);
  }

  public registerNeuron(
    docId: string,
    termFreqs: Map<string, number>,
    docLength: number,
    links?: readonly string[]
  ): void {
    this.docCount++;
    this.totalDocLength += docLength;

    for (const [token, tf] of termFreqs.entries()) {
      let postings = this.invertedIndex.get(token);
      if (!postings) {
        postings = new Map<string, number>();
        this.invertedIndex.set(token, postings);
      }
      postings.set(docId, tf);
    }

    if (links && links.length > 0) {
      let outSet = this.docGraph.get(docId);
      if (!outSet) {
        outSet = new Set<string>();
        this.docGraph.set(docId, outSet);
      }
      for (const targetId of links) {
        if (!targetId || targetId === docId) continue;
        outSet.add(targetId);
        let inSet = this.docGraph.get(targetId);
        if (!inSet) {
          inSet = new Set<string>();
          this.docGraph.set(targetId, inSet);
        }
        inSet.add(docId);
      }
    }
  }

  public unregisterNeuron(docId: string, termFreqs: Map<string, number>, docLength: number): void {
    this.docCount = Math.max(0, this.docCount - 1);
    this.totalDocLength = Math.max(0, this.totalDocLength - docLength);

    for (const token of termFreqs.keys()) {
      const postings = this.invertedIndex.get(token);
      if (!postings) continue;
      postings.delete(docId);
      if (postings.size === 0) {
        this.invertedIndex.delete(token);
      }
    }

    const neighbors = this.docGraph.get(docId);
    if (neighbors) {
      for (const n of neighbors) {
        this.docGraph.get(n)?.delete(docId);
      }
      this.docGraph.delete(docId);
    }
  }

  /**
   * 이벤트 구동 LIF 스파이크 시뮬레이션 실행
   */
  public simulateSpikes(
    queryTokens: readonly string[],
    docLengths: ReadonlyMap<string, number>,
    basalCurrent?: ReadonlyMap<string, number>
  ): {
    readonly activations: readonly SnnActivationState[];
    readonly activeNeuronsEvaluated: number;
    readonly firedNeurons: number;
  } {
    if ((queryTokens.length === 0 && (!basalCurrent || basalCurrent.size === 0)) || this.docCount === 0) {
      return { activations: [], activeNeuronsEvaluated: 0, firedNeurons: 0 };
    }

    const avgDocLen = this.totalDocLength / Math.max(1, this.docCount);
    const uniqueQueryTokens = Array.from(new Set(queryTokens));

    // 1. 역색인 기반 시냅스 입력 전류(I_ext) 계산 (관련 뉴런만 O(active)로 수집)
    const inputCurrent = new Map<string, number>();
    const rawBm25 = new Map<string, number>();

    for (const token of uniqueQueryTokens) {
      const postings = this.invertedIndex.get(token);
      if (!postings || postings.size === 0) continue;

      const df = postings.size;
      // BM25 IDF
      const idf = Math.log(1 + (this.docCount - df + 0.5) / (df + 0.5));
      const isNgram = token.startsWith('~');
      const tokenFactor = isNgram ? 0.45 : 1.0;

      for (const [docId, tf] of postings.entries()) {
        const dl = docLengths.get(docId) ?? avgDocLen;
        const bm25Term =
          idf *
          ((tf * (SNN_RAG_CONFIG.DEFAULT_BM25_K1 + 1)) /
            (tf +
              SNN_RAG_CONFIG.DEFAULT_BM25_K1 *
                (1 - SNN_RAG_CONFIG.DEFAULT_BM25_B + SNN_RAG_CONFIG.DEFAULT_BM25_B * (dl / Math.max(1, avgDocLen)))));

        const synKey = `${token}\0${docId}`;
        const stdpWeight = this.synapseWeights.get(synKey) ?? 1.0;

        rawBm25.set(docId, (rawBm25.get(docId) ?? 0) + bm25Term * tokenFactor);
        inputCurrent.set(docId, (inputCurrent.get(docId) ?? 0) + bm25Term * tokenFactor * stdpWeight);
      }
    }

    // STDP로 학습된 직접 시냅스(동의어/연관어)도 역색인에 없던 문서로 전류 전달
    for (const token of uniqueQueryTokens) {
      if (token.startsWith('~')) continue;
      const prefix = `${token}\0`;
      for (const [synKey, weight] of this.synapseWeights.entries()) {
        if (!synKey.startsWith(prefix) || weight <= 1.05) continue;
        const docId = synKey.slice(prefix.length);
        if (!docLengths.has(docId)) continue;
        if (!inputCurrent.has(docId)) {
          inputCurrent.set(docId, (weight - 1.0) * 1.4);
        }
      }
    }

    // 수상돌기 Dense 기저 전류가 있는 후보 뉴런도 활성 집합에 등록
    if (basalCurrent && basalCurrent.size > 0) {
      for (const docId of basalCurrent.keys()) {
        if (docLengths.has(docId) && !inputCurrent.has(docId)) {
          inputCurrent.set(docId, 0);
        }
      }
    }

    if (inputCurrent.size === 0) {
      return { activations: [], activeNeuronsEvaluated: 0, firedNeurons: 0 };
    }

    // 최대 전류 기준으로 정규화하여 스케일 불변성 확보
    let maxCurrent = 0;
    let maxBm25 = 0;
    for (const val of inputCurrent.values()) {
      if (val > maxCurrent) maxCurrent = val;
    }
    for (const val of rawBm25.values()) {
      if (val > maxBm25) maxBm25 = val;
    }
    const normCurrentDivisor = maxCurrent > 0 ? maxCurrent : 1;
    const normBm25Divisor = maxBm25 > 0 ? maxBm25 : 1;

    // 2. LIF 뉴런 상태 초기화 (활성 뉴런만 메모리에 생성)
    const potential = new Map<string, number>();
    const refractory = new Map<string, number>();
    const spikeCount = new Map<string, number>();
    const firstSpikeStep = new Map<string, number>();
    const peakPotential = new Map<string, number>();
    const hopDistance = new Map<string, number>();

    for (const docId of inputCurrent.keys()) {
      potential.set(docId, 0);
      refractory.set(docId, 0);
      spikeCount.set(docId, 0);
      firstSpikeStep.set(docId, -1);
      peakPotential.set(docId, 0);
      hopDistance.set(docId, 0);
    }

    // 3. 다중 타임스텝 LIF 시뮬레이션 + 측면 억제 + 다중 홉 시냅스 전파
    for (let step = 1; step <= this.timeSteps; step++) {
      const firedThisStep: Array<{ docId: string; excess: number }> = [];

      // (a) 누설 적분(Leaky Integration) 및 발화 판정
      for (const [docId, curr] of inputCurrent.entries()) {
        const ref = refractory.get(docId) ?? 0;
        if (ref > 0) {
          refractory.set(docId, ref - 1);
          continue;
        }
        const lexicalI = maxCurrent > 0 ? (curr / normCurrentDivisor) * 0.65 : 0;
        const dendriticI = basalCurrent ? (basalCurrent.get(docId) ?? 0) * SNN_RAG_CONFIG.DEFAULT_DENDRITIC_GAIN : 0;
        const normalizedI = Math.min(0.95, lexicalI + dendriticI);
        const nextV = (potential.get(docId) ?? 0) * this.leakFactor + normalizedI;
        if (nextV > (peakPotential.get(docId) ?? 0)) {
          peakPotential.set(docId, nextV);
        }

        if (nextV >= this.firingThreshold) {
          spikeCount.set(docId, (spikeCount.get(docId) ?? 0) + 1);
          if ((firstSpikeStep.get(docId) ?? -1) === -1) {
            firstSpikeStep.set(docId, step);
          }
          // 소프트 리셋(Soft Reset)
          potential.set(docId, nextV - this.firingThreshold);
          refractory.set(docId, this.refractorySteps);
          firedThisStep.push({ docId, excess: nextV - this.firingThreshold });
        } else {
          potential.set(docId, nextV);
        }
      }

      if (firedThisStep.length === 0) continue;

      // (b) 측면 억제(Lateral Inhibition): 최상위 발화 뉴런이 비발화 경쟁 뉴런의 전위를 억제
      if (this.lateralInhibition > 0) {
        const firedSet = new Set(firedThisStep.map((f) => f.docId));
        const inhibitionPulse = this.lateralInhibition * Math.min(1, firedThisStep.length / 4);
        for (const [docId, v] of potential.entries()) {
          if (!firedSet.has(docId) && v > 0) {
            potential.set(docId, Math.max(0, v - inhibitionPulse));
          }
        }
      }

      // (c) 다중 홉 시냅스 전파(Spreading Activation): 발화 뉴런이 연결된 이웃 뉴런에 스파이크 전류 전달
      if (this.maxHops > 0 && this.hopDecay > 0) {
        for (const { docId } of firedThisStep) {
          const currentHop = hopDistance.get(docId) ?? 0;
          if (currentHop >= this.maxHops) continue;

          const neighbors = this.docGraph.get(docId);
          if (!neighbors || neighbors.size === 0) continue;

          const propagatedCurrent = (inputCurrent.get(docId) ?? 0) * this.hopDecay;
          if (propagatedCurrent <= 0.01) continue;

          for (const neighborId of neighbors) {
            if (!docLengths.has(neighborId)) continue;
            const prevI = inputCurrent.get(neighborId) ?? 0;
            // 이웃 뉴런에 시냅스 전류 주입
            inputCurrent.set(neighborId, prevI + propagatedCurrent * 0.5);
            if (!potential.has(neighborId)) {
              potential.set(neighborId, 0);
              refractory.set(neighborId, 0);
              spikeCount.set(neighborId, 0);
              firstSpikeStep.set(neighborId, -1);
              peakPotential.set(neighborId, 0);
              hopDistance.set(neighborId, currentHop + 1);
            }
          }
        }
      }
    }

    // 4. 발화 결과 집계 (Rate Coding + Time-to-First-Spike Latency Bonus)
    const activations: SnnActivationState[] = [];
    let firedNeurons = 0;

    for (const [docId, spikes] of spikeCount.entries()) {
      const firstStep = firstSpikeStep.get(docId) ?? -1;
      const peakV = peakPotential.get(docId) ?? 0;
      const hops = hopDistance.get(docId) ?? 0;
      const bm25Norm = (rawBm25.get(docId) ?? 0) / normBm25Divisor;

      if (spikes > 0) {
        firedNeurons++;
        const rateScore = Math.min(1, spikes / Math.max(1, this.timeSteps * 0.65));
        const latencyBonus = (this.timeSteps - firstStep + 1) / this.timeSteps;
        const hopPenalty = Math.pow(0.88, hops);
        const snnScore = Math.min(1, (rateScore * 0.6 + latencyBonus * 0.4) * hopPenalty);

        activations.push({
          docId,
          spikeCount: spikes,
          firstSpikeStep: firstStep,
          peakPotential: Number(peakV.toFixed(4)),
          hopDistance: hops,
          snnScore: Number(snnScore.toFixed(4)),
          bm25Score: Number(bm25Norm.toFixed(4)),
        });
      }
    }

    // 만약 어떤 뉴런도 임계치를 넘지 못했다면(극단적 희소 쿼리), 최고 막전위 상위 후보를 서브-임계치로 보존
    if (activations.length === 0) {
      const subThreshold = Array.from(peakPotential.entries())
        .sort((a, b) => b[1] - a[1])
        .slice(0, SNN_RAG_CONFIG.DEFAULT_MIN_STAGE1_CANDIDATES);

      for (const [docId, peakV] of subThreshold) {
        if (peakV <= 0) continue;
        const bm25Norm = (rawBm25.get(docId) ?? 0) / normBm25Divisor;
        activations.push({
          docId,
          spikeCount: 0,
          firstSpikeStep: -1,
          peakPotential: Number(peakV.toFixed(4)),
          hopDistance: hopDistance.get(docId) ?? 0,
          snnScore: Number(Math.min(0.49, peakV / Math.max(0.01, this.firingThreshold) * 0.45).toFixed(4)),
          bm25Score: Number(bm25Norm.toFixed(4)),
        });
      }
    }

    activations.sort((a, b) => b.snnScore - a.snnScore || b.bm25Score - a.bm25Score);

    return {
      activations,
      activeNeuronsEvaluated: potential.size,
      firedNeurons,
    };
  }

  /**
   * STDP (Spike-Timing-Dependent Plasticity) 온라인 시냅스 가중치 갱신
   * 쿼리-채택 문서 쌍은 장기 강화(LTP), 쿼리-오탐 문서 쌍은 장기 약화(LTD)
   */
  public applyStdpFeedback(
    queryTokens: readonly string[],
    relevantDocIds: readonly string[],
    irrelevantDocIds: readonly string[] = []
  ): StdpFeedbackResult {
    const coreTokens = Array.from(new Set(queryTokens.filter((t) => !t.startsWith('~'))));
    let updatedSynapses = 0;

    for (const docId of relevantDocIds) {
      for (const token of coreTokens) {
        const key = `${token}\0${docId}`;
        const prev = this.synapseWeights.get(key) ?? 1.0;
        const next = Math.min(SNN_RAG_CONFIG.DEFAULT_MAX_SYNAPSE_WEIGHT, prev + this.stdpLtpRate);
        this.synapseWeights.set(key, Number(next.toFixed(4)));
        updatedSynapses++;
      }
    }

    for (const docId of irrelevantDocIds) {
      for (const token of coreTokens) {
        const key = `${token}\0${docId}`;
        const prev = this.synapseWeights.get(key) ?? 1.0;
        const next = Math.max(SNN_RAG_CONFIG.DEFAULT_MIN_SYNAPSE_WEIGHT, prev - this.stdpLtdRate);
        this.synapseWeights.set(key, Number(next.toFixed(4)));
        updatedSynapses++;
      }
    }

    return {
      updatedSynapses,
      strengthenedDocs: relevantDocIds.length,
      weakenedDocs: irrelevantDocIds.length,
    };
  }

  public getSynapseWeight(token: string, docId: string): number {
    return this.synapseWeights.get(`${token.toLowerCase()}\0${docId}`) ?? 1.0;
  }

  public getSynapseCount(): number {
    return this.synapseWeights.size;
  }
}

/**
 * SNN-RAG 하이브리드 검색 엔진
 * 1단계 Event-Driven SNN 게이팅 + 2단계 플러그형 Neural Dense/Reranker 융합 + Active Inference 토큰 예산 선택
 */
export class HybridSnnRagEngine {
  private readonly snnLayer: SnnSynapticLayer;
  private readonly docs = new Map<string, IndexedNeuronDoc>();
  private readonly docLengths = new Map<string, number>();
  private readonly neuralEmbedder: VectorEmbedderFn;
  private readonly queryEmbedder: VectorEmbedderFn;
  private readonly neuralReranker?: NeuralRerankerFn;
  private readonly weights: { SNN: number; BM25: number; DENSE: number; RERANK: number };
  private readonly aifSelector = new ActiveInferenceContextSelector();
  private hasPrecomputedEmbeddings = false;

  constructor(options?: SnnRagOptions) {
    this.snnLayer = new SnnSynapticLayer(options);
    this.neuralEmbedder = options?.neuralEmbedder ?? ((txt) => localHashEmbed(txt));
    this.queryEmbedder = options?.queryEmbedder ?? this.neuralEmbedder;
    this.neuralReranker = options?.neuralReranker;
    this.weights = {
      SNN: options?.weights?.SNN ?? SNN_RAG_CONFIG.WEIGHTS.SNN,
      BM25: options?.weights?.BM25 ?? SNN_RAG_CONFIG.WEIGHTS.BM25,
      DENSE: options?.weights?.DENSE ?? SNN_RAG_CONFIG.WEIGHTS.DENSE,
      RERANK: options?.weights?.RERANK ?? SNN_RAG_CONFIG.WEIGHTS.RERANK,
    };
  }

  /**
   * 코퍼스 문서를 사전 배치 임베딩하여 수상돌기 Dense 기저 전류(Dendritic Mode) 활성화
   */
  public async precomputeEmbeddings(
    batchEmbedder?: (texts: readonly string[]) => Promise<number[][]>
  ): Promise<number> {
    const pending: IndexedNeuronDoc[] = [];
    for (const item of this.docs.values()) {
      if (!item.cachedDenseVector || !item.cachedDenseNorm) {
        pending.push(item);
      }
    }
    if (pending.length === 0) {
      this.hasPrecomputedEmbeddings = this.docs.size > 0;
      return 0;
    }

    const texts = pending.map((p) => (p.doc.title ? `${p.doc.title}\n${p.doc.text}` : p.doc.text));
    if (batchEmbedder) {
      const vectors = await batchEmbedder(texts);
      for (let i = 0; i < pending.length; i++) {
        const v = toFloat32AndNorm(vectors[i] ?? []);
        pending[i].cachedDenseVector = v.arr;
        pending[i].cachedDenseNorm = v.norm;
      }
    } else {
      for (let i = 0; i < pending.length; i++) {
        const raw = await this.neuralEmbedder(texts[i]);
        const v = toFloat32AndNorm(raw);
        pending[i].cachedDenseVector = v.arr;
        pending[i].cachedDenseNorm = v.norm;
      }
    }

    this.hasPrecomputedEmbeddings = true;
    return pending.length;
  }

  /**
   * 문서(뉴런) 단건 등록 또는 갱신
   */
  public upsert(doc: SnnRagDocument): void {
    if (!doc || !doc.id || typeof doc.text !== 'string') {
      throw new Error('유효한 문서 ID와 텍스트가 필요합니다.');
    }

    const existing = this.docs.get(doc.id);
    if (existing) {
      this.snnLayer.unregisterNeuron(doc.id, existing.termFreqs, existing.docLength);
    }

    const compositeText = [doc.title ?? '', doc.text, ...(doc.tags ?? [])].join(' ');
    const tokens = tokenizeForSnn(compositeText);
    const termFreqs = new Map<string, number>();
    for (const tok of tokens) {
      termFreqs.set(tok, (termFreqs.get(tok) ?? 0) + 1);
    }

    const indexed: IndexedNeuronDoc = {
      doc,
      tokens,
      termFreqs,
      docLength: Math.max(1, tokens.length),
      tokenCost: estimateTokenCost(doc.text, doc.tokenCost),
    };

    this.docs.set(doc.id, indexed);
    this.docLengths.set(doc.id, indexed.docLength);
    this.snnLayer.registerNeuron(doc.id, termFreqs, indexed.docLength, doc.links);
  }

  /**
   * 문서 일괄 등록
   */
  public upsertBatch(docs: readonly SnnRagDocument[]): number {
    let count = 0;
    for (const doc of docs) {
      this.upsert(doc);
      count++;
    }
    return count;
  }

  /**
   * 문서 삭제
   */
  public remove(docId: string): boolean {
    const existing = this.docs.get(docId);
    if (!existing) return false;
    this.snnLayer.unregisterNeuron(docId, existing.termFreqs, existing.docLength);
    this.docs.delete(docId);
    this.docLengths.delete(docId);
    return true;
  }

  public size(): number {
    return this.docs.size;
  }

  /**
   * STDP 피드백 기록 (온라인 시냅스 강화/약화)
   */
  public recordFeedback(
    query: string,
    relevantDocIds: readonly string[],
    irrelevantDocIds: readonly string[] = []
  ): StdpFeedbackResult {
    const queryTokens = tokenizeForSnn(query);
    return this.snnLayer.applyStdpFeedback(queryTokens, relevantDocIds, irrelevantDocIds);
  }

  public getSynapseWeight(token: string, docId: string): number {
    return this.snnLayer.getSynapseWeight(token, docId);
  }

  /**
   * SNN-RAG 2단계 하이브리드 검색 실행
   */
  public async search(query: string, options?: SnnRagQueryOptions): Promise<SnnRagSearchResult> {
    const tTotal0 = performance.now();
    const topK = Math.max(1, options?.topK ?? SNN_RAG_CONFIG.DEFAULT_TOP_K);
    const totalDocuments = this.docs.size;

    if (!query || !query.trim() || totalDocuments === 0) {
      return {
        hits: [],
        telemetry: {
          totalDocuments,
          activeNeuronsEvaluated: 0,
          firedNeurons: 0,
          spikeSparsity: 1,
          neuralCandidatesEvaluated: 0,
          neuralComputeSavedRatio: 1,
          stage1SnnMs: 0,
          stage2NeuralMs: 0,
          totalMs: Number((performance.now() - tTotal0).toFixed(3)),
          fallbackTriggered: false,
        },
      };
    }

    const denseScoreMap = new Map<string, number>();
    const rerankScoreMap = new Map<string, number>();
    let fallbackTriggered = false;
    let warningMessage: string | undefined;
    let cachedQueryVec: { arr: Float32Array; norm: number } | undefined;

    // 수상돌기 Dense 기저 전류 계산 (사전 임베딩 활성화 시)
    let basalCurrent: Map<string, number> | undefined;
    const useDendritic =
      !options?.pureSnnOnly && (options?.enableDendriticCurrent ?? true) && this.hasPrecomputedEmbeddings;

    const tSnn0 = performance.now();
    if (useDendritic) {
      try {
        const qRaw = await this.queryEmbedder(query);
        cachedQueryVec = toFloat32AndNorm(qRaw);
        const scoredDense: Array<{ id: string; sim: number }> = [];

        for (const [id, item] of this.docs.entries()) {
          if (!item.cachedDenseVector || !item.cachedDenseNorm) continue;
          const simScore = cosineSimilarityFloat32(
            cachedQueryVec.arr,
            cachedQueryVec.norm,
            item.cachedDenseVector,
            item.cachedDenseNorm
          );
          const rounded = Number(simScore.toFixed(4));
          denseScoreMap.set(id, rounded);
          scoredDense.push({ id, sim: rounded });
        }

        scoredDense.sort((a, b) => b.sim - a.sim);
        const topDendritic = scoredDense.slice(0, SNN_RAG_CONFIG.DEFAULT_DENDRITIC_TOP_K);
        if (topDendritic.length > 0) {
          const maxSim = topDendritic[0].sim;
          const minSim = topDendritic[topDendritic.length - 1].sim;
          const span = Math.max(0.0001, maxSim - minSim);
          basalCurrent = new Map<string, number>();
          for (const d of topDendritic) {
            const rel = topDendritic.length === 1 ? 1 : (d.sim - minSim) / span;
            basalCurrent.set(d.id, Number((0.35 + rel * 0.65).toFixed(4)));
          }
        }
      } catch (err) {
        fallbackTriggered = true;
        warningMessage = `수상돌기 쿼리 임베딩 실패로 어휘 SNN 단독 진행: ${
          err instanceof Error ? err.message : String(err)
        }`;
      }
    }

    // --- 1단계: Event-Driven SNN LIF 스파이크 게이팅 ---
    const queryTokens = tokenizeForSnn(query);
    const sim = this.snnLayer.simulateSpikes(queryTokens, this.docLengths, basalCurrent);
    const stage1SnnMs = performance.now() - tSnn0;

    const maxNeuralCandidates = Math.max(
      topK,
      options?.maxNeuralCandidates ??
        Math.max(SNN_RAG_CONFIG.DEFAULT_MIN_STAGE1_CANDIDATES, topK * SNN_RAG_CONFIG.DEFAULT_STAGE1_MULTIPLIER)
    );
    const stage1Candidates = sim.activations.slice(0, maxNeuralCandidates);

    // --- 2단계: 발화 후보에 한해 Neural Dense 임베딩 + Neural Reranker 선택적 실행 ---
    const tNeural0 = performance.now();

    if (!options?.pureSnnOnly && stage1Candidates.length > 0) {
      // 2-A. Dense 임베딩 유사도 계산 (지연 캐싱 + 장애 시 로컬 해시 임베더 폴백)
      try {
        const qVec = cachedQueryVec ?? toFloat32AndNorm(await this.queryEmbedder(query));

        for (const cand of stage1Candidates) {
          if (denseScoreMap.has(cand.docId)) continue;
          const item = this.docs.get(cand.docId);
          if (!item) continue;
          if (!item.cachedDenseVector || !item.cachedDenseNorm) {
            const dRaw = await this.neuralEmbedder(
              item.doc.title ? `${item.doc.title}\n${item.doc.text}` : item.doc.text
            );
            const dVec = toFloat32AndNorm(dRaw);
            item.cachedDenseVector = dVec.arr;
            item.cachedDenseNorm = dVec.norm;
          }
          const simScore = cosineSimilarityFloat32(
            qVec.arr,
            qVec.norm,
            item.cachedDenseVector,
            item.cachedDenseNorm
          );
          denseScoreMap.set(cand.docId, Number(simScore.toFixed(4)));
        }
      } catch (err) {
        fallbackTriggered = true;
        warningMessage = `신경망 임베더 호출 실패로 로컬 폴백 전환: ${
          err instanceof Error ? err.message : String(err)
        }`;
        const qFallback = toFloat32AndNorm(localHashEmbed(query));
        for (const cand of stage1Candidates) {
          const item = this.docs.get(cand.docId);
          if (!item) continue;
          const dFallback = toFloat32AndNorm(localHashEmbed(item.doc.text));
          const simScore = cosineSimilarityFloat32(
            qFallback.arr,
            qFallback.norm,
            dFallback.arr,
            dFallback.norm
          );
          denseScoreMap.set(cand.docId, Number(simScore.toFixed(4)));
        }
      }

      // 2-B. 선택적 신경망 리랭커 호출 (장애 시 무중단 폴백)
      if (this.neuralReranker) {
        try {
          const rerankInput: NeuralRerankCandidate[] = [];
          for (const cand of stage1Candidates) {
            const item = this.docs.get(cand.docId);
            if (!item) continue;
            rerankInput.push({
              id: cand.docId,
              text: item.doc.text,
              title: item.doc.title,
              snnScore: cand.snnScore,
              denseScore: denseScoreMap.get(cand.docId) ?? 0,
            });
          }
          const reranked = await this.neuralReranker(query, rerankInput);
          if (reranked instanceof Map) {
            for (const [id, score] of reranked.entries()) {
              rerankScoreMap.set(id, Math.max(0, Math.min(1, Number(score) || 0)));
            }
          } else if (reranked && typeof reranked === 'object') {
            for (const [id, score] of Object.entries(reranked)) {
              rerankScoreMap.set(id, Math.max(0, Math.min(1, Number(score) || 0)));
            }
          }
        } catch (err) {
          fallbackTriggered = true;
          const msg = `신경망 리랭커 호출 실패로 SNN+Dense 폴백 전환: ${
            err instanceof Error ? err.message : String(err)
          }`;
          warningMessage = warningMessage ? `${warningMessage} | ${msg}` : msg;
        }
      }
    }
    const stage2NeuralMs = performance.now() - tNeural0;

    // --- 3단계: 하이브리드 점수 융합 ---
    const hasRerank = !options?.pureSnnOnly && rerankScoreMap.size > 0;
    const hasDense = !options?.pureSnnOnly && denseScoreMap.size > 0;

    const wSnn = this.weights.SNN;
    const wBm25 = this.weights.BM25;
    const wDense = hasDense ? this.weights.DENSE : 0;
    const wRerank = hasRerank ? this.weights.RERANK : 0;
    const wSum = Math.max(0.0001, wSnn + wBm25 + wDense + wRerank);

    const hits: SnnRagHit[] = [];
    for (const cand of stage1Candidates) {
      const item = this.docs.get(cand.docId);
      if (!item) continue;
      const denseScore = denseScoreMap.get(cand.docId) ?? 0;
      const rerankScore = rerankScoreMap.get(cand.docId) ?? 0;
      const fused =
        (cand.snnScore * wSnn +
          cand.bm25Score * wBm25 +
          denseScore * wDense +
          rerankScore * wRerank) /
        wSum;

      hits.push({
        id: cand.docId,
        score: Number(fused.toFixed(4)),
        snnScore: cand.snnScore,
        bm25Score: cand.bm25Score,
        denseScore,
        rerankScore,
        spikeCount: cand.spikeCount,
        firstSpikeStep: cand.firstSpikeStep,
        hopDistance: cand.hopDistance,
        tokenCost: item.tokenCost,
        title: item.doc.title,
        text: item.doc.text,
        metadata: item.doc.metadata,
      });
    }

    hits.sort((a, b) => b.score - a.score || b.spikeCount - a.spikeCount);
    const finalHits = hits.slice(0, topK);

    // --- 4단계 (선택): Active Inference 토큰 예산 최적화 ---
    let selectedByBudgetIds: readonly string[] | undefined;
    let totalSelectedTokens: number | undefined;
    if (typeof options?.tokenBudget === 'number' && options.tokenBudget > 0 && finalHits.length > 0) {
      const aifCandidates: ContextCandidate[] = finalHits.map((h) => ({
        id: h.id,
        tokenCost: h.tokenCost,
        relevance: h.score,
        // 다중 홉으로 발견된 문서나 높은 리랭크 점수 문서는 불확실성 해소(epistemicGain) 가점
        epistemicGain: Math.min(1, h.score * 0.7 + (h.hopDistance > 0 ? 0.25 : 0.15)),
      }));
      const aifRes = this.aifSelector.select(aifCandidates, options.tokenBudget);
      selectedByBudgetIds = aifRes.selectedIds;
      totalSelectedTokens = aifRes.totalTokens;
    }

    const neuralEvaluated = options?.pureSnnOnly ? 0 : stage1Candidates.length;
    const spikeSparsity = Number(
      Math.max(0, Math.min(1, 1 - sim.firedNeurons / Math.max(1, totalDocuments))).toFixed(4)
    );
    const neuralComputeSavedRatio = Number(
      Math.max(0, Math.min(1, 1 - neuralEvaluated / Math.max(1, totalDocuments))).toFixed(4)
    );

    return {
      hits: finalHits,
      selectedByBudgetIds,
      totalSelectedTokens,
      telemetry: {
        totalDocuments,
        activeNeuronsEvaluated: sim.activeNeuronsEvaluated,
        firedNeurons: sim.firedNeurons,
        spikeSparsity,
        neuralCandidatesEvaluated: neuralEvaluated,
        neuralComputeSavedRatio,
        stage1SnnMs: Number(stage1SnnMs.toFixed(3)),
        stage2NeuralMs: Number(stage2NeuralMs.toFixed(3)),
        totalMs: Number((performance.now() - tTotal0).toFixed(3)),
        fallbackTriggered,
        warningMessage,
      },
    };
  }
}
