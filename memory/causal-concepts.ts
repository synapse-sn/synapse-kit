/**
 * 시냅스 초학제적 핵심 개념 구현 엔진 (Interdisciplinary Cognitive & Control Concepts)
 * 1. 능동 추론 (Active Inference / Expected Free Energy) 컨텍스트 선택기
 * 2. 항상성 PID 피드백 제어기 (Homeostatic Control)
 * 3. 인공 면역 체계 (Artificial Immune System: Negative Selection & Clonal Memory)
 * 4. 최소 기술 길이 (Minimum Description Length, MDL) 그래프 증류 및 이상 탐지
 * 5. 게임이론 비례 배분 경매 (Kelly Proportional Allocation Mechanism)
 * 6. 검색 증강 이상 탐지 (RAAD: Retrieval-Augmented Anomaly Detection)
 */

export const INTERDISCIPLINARY_CONFIG = {
  DEFAULT_TOKEN_PENALTY: 0.002,
  DEFAULT_KP: 0.6,
  DEFAULT_KI: 0.15,
  DEFAULT_KD: 0.1,
  DEFAULT_SELF_RADIUS: 0.25,
  DEFAULT_CLONAL_TTL_MS: 3_600_000,
  DEFAULT_MDL_RULE_BITS: 32,
  DEFAULT_MDL_UNCOVERED_BITS: 16,
  DEFAULT_RAAD_ANOMALY_THRESHOLD: 0.65,
} as const;

export interface ContextCandidate {
  readonly id: string;
  readonly tokenCost: number;
  readonly relevance: number; // 0..1
  readonly epistemicGain: number; // 0..1 (불확실성 해소량)
}

export interface ActiveInferenceResult {
  readonly selectedIds: readonly string[];
  readonly totalTokens: number;
  readonly totalFreeEnergyReduction: number;
}

/**
 * 1. 능동 추론(Active Inference / FEP) 기반 컨텍스트 선택기
 * 토큰 비용 대비 예상 자유 에너지(불확실성) 감소량이 가장 큰 컨텍스트 조합을 예산 내에서 선택합니다.
 */
export class ActiveInferenceContextSelector {
  constructor(
    private readonly tokenPenalty: number = INTERDISCIPLINARY_CONFIG.DEFAULT_TOKEN_PENALTY
  ) {}

  public select(
    candidates: readonly ContextCandidate[],
    tokenBudget: number
  ): ActiveInferenceResult {
    if (tokenBudget <= 0 || candidates.length === 0) {
      return { selectedIds: [], totalTokens: 0, totalFreeEnergyReduction: 0 };
    }

    const scored = candidates
      .map((c) => {
        const efeReduction = c.epistemicGain * 0.6 + c.relevance * 0.4 - this.tokenPenalty * c.tokenCost;
        const density = c.tokenCost > 0 ? efeReduction / c.tokenCost : efeReduction;
        return { candidate: c, efeReduction, density };
      })
      .filter((x) => x.efeReduction > 0 && x.candidate.tokenCost <= tokenBudget)
      .sort((a, b) => b.density - a.density);

    const selectedIds: string[] = [];
    let totalTokens = 0;
    let totalFreeEnergyReduction = 0;

    for (const item of scored) {
      if (totalTokens + item.candidate.tokenCost <= tokenBudget) {
        selectedIds.push(item.candidate.id);
        totalTokens += item.candidate.tokenCost;
        totalFreeEnergyReduction += item.efeReduction;
      }
    }

    return {
      selectedIds,
      totalTokens,
      totalFreeEnergyReduction: Number(totalFreeEnergyReduction.toFixed(4)),
    };
  }
}

export interface HomeostaticOutput {
  readonly error: number;
  readonly controlSignal: number; // -1..1 정규화 제어량
  readonly recommendedForgettingRate: number; // 0.01..0.5
}

/**
 * 2. 항상성 PID 피드백 제어기 (Homeostatic Control)
 * 목표 세트포인트(지연·메모리·부하 등)를 유지하기 위해 망각률/배압 제어량을 동적 산출합니다.
 */
export class HomeostaticController {
  private integral = 0;
  private prevError = 0;

  constructor(
    private readonly setpoint: number,
    private readonly kp: number = INTERDISCIPLINARY_CONFIG.DEFAULT_KP,
    private readonly ki: number = INTERDISCIPLINARY_CONFIG.DEFAULT_KI,
    private readonly kd: number = INTERDISCIPLINARY_CONFIG.DEFAULT_KD
  ) {}

  public step(measured: number, dt: number = 1): HomeostaticOutput {
    const safeDt = dt > 0 ? dt : 1;
    const error = measured - this.setpoint;
    this.integral = Math.max(-10, Math.min(10, this.integral + error * safeDt));
    const derivative = (error - this.prevError) / safeDt;
    this.prevError = error;

    const raw = this.kp * error + this.ki * this.integral + this.kd * derivative;
    const controlSignal = Math.max(-1, Math.min(1, raw / Math.max(1, this.setpoint)));
    const recommendedForgettingRate = Number(
      Math.max(0.01, Math.min(0.5, 0.05 + Math.max(0, controlSignal) * 0.45)).toFixed(4)
    );

    return {
      error: Number(error.toFixed(4)),
      controlSignal: Number(controlSignal.toFixed(4)),
      recommendedForgettingRate,
    };
  }

  public reset(): void {
    this.integral = 0;
    this.prevError = 0;
  }
}

export interface ImmuneEvaluation {
  readonly isAnomaly: boolean;
  readonly minSelfDistance: number;
  readonly matchedCloneId: string | null;
}

/**
 * 3. 인공 면역 체계 (Negative Selection + Clonal Memory)
 * 정상(Self) 공간에서 벗어난 미지 패턴(Non-Self)을 탐지하고 확정된 위협을 클론 메모리에 각인합니다.
 */
export class ImmuneAnomalyDetector {
  private readonly selfSamples: number[][] = [];
  private readonly clonalMemory = new Map<string, { vector: number[]; createdAt: number }>();

  constructor(private readonly selfRadius: number = INTERDISCIPLINARY_CONFIG.DEFAULT_SELF_RADIUS) {}

  public trainSelf(vector: readonly number[]): void {
    this.selfSamples.push([...vector]);
  }

  public cloneThreat(id: string, vector: readonly number[]): void {
    this.clonalMemory.set(id, { vector: [...vector], createdAt: Date.now() });
  }

  public evaluate(vector: readonly number[]): ImmuneEvaluation {
    // 1. 클론 기억(기존 위협 항체) 우선 대조
    for (const [id, entry] of this.clonalMemory.entries()) {
      if (this.euclideanDistance(vector, entry.vector) <= this.selfRadius * 0.8) {
        return { isAnomaly: true, minSelfDistance: Infinity, matchedCloneId: id };
      }
    }

    if (this.selfSamples.length === 0) {
      return { isAnomaly: false, minSelfDistance: 0, matchedCloneId: null };
    }

    // 2. Negative Selection: 모든 Self 샘플과의 최소 거리 계산
    let minDist = Infinity;
    for (const sample of this.selfSamples) {
      const dist = this.euclideanDistance(vector, sample);
      if (dist < minDist) minDist = dist;
    }

    return {
      isAnomaly: minDist > this.selfRadius,
      minSelfDistance: Number(minDist.toFixed(4)),
      matchedCloneId: null,
    };
  }

  private euclideanDistance(a: readonly number[], b: readonly number[]): number {
    const len = Math.min(a.length, b.length);
    let sum = 0;
    for (let i = 0; i < len; i++) {
      const diff = a[i] - b[i];
      sum += diff * diff;
    }
    return Math.sqrt(sum);
  }
}

export interface MdlRuleItem {
  readonly id: string;
  readonly covers: readonly string[];
}

export interface MdlDistillResult {
  readonly retainedIds: readonly string[];
  readonly totalDescriptionBits: number;
  readonly compressionSavingsBits: number;
}

/**
 * 4. 최소 기술 길이(MDL) 기반 규칙 집합 증류기
 * 동일 데이터를 가장 짧은 모델+잔차 설명 길이로 표현하는 최소 규칙 집합을 추출합니다.
 */
export class MdlGraphDistiller {
  constructor(
    private readonly ruleCostBits: number = INTERDISCIPLINARY_CONFIG.DEFAULT_MDL_RULE_BITS,
    private readonly uncoveredCostBits: number = INTERDISCIPLINARY_CONFIG.DEFAULT_MDL_UNCOVERED_BITS
  ) {}

  public distill(
    rules: readonly MdlRuleItem[],
    universeFacts: readonly string[]
  ): MdlDistillResult {
    const uncovered = new Set(universeFacts);
    const remaining = [...rules];
    const retainedIds: string[] = [];

    const initialBits =
      rules.length * this.ruleCostBits +
      this.countUncovered(rules, universeFacts) * this.uncoveredCostBits;

    while (uncovered.size > 0 && remaining.length > 0) {
      let bestIdx = -1;
      let bestGain = 0;

      for (let i = 0; i < remaining.length; i++) {
        let newCovered = 0;
        for (const fact of remaining[i].covers) {
          if (uncovered.has(fact)) newCovered++;
        }
        const netBitGain = newCovered * this.uncoveredCostBits - this.ruleCostBits;
        if (netBitGain > bestGain) {
          bestGain = netBitGain;
          bestIdx = i;
        }
      }

      if (bestIdx === -1) break;
      const chosen = remaining.splice(bestIdx, 1)[0];
      retainedIds.push(chosen.id);
      for (const fact of chosen.covers) {
        uncovered.delete(fact);
      }
    }

    const totalDescriptionBits =
      retainedIds.length * this.ruleCostBits + uncovered.size * this.uncoveredCostBits;

    return {
      retainedIds,
      totalDescriptionBits,
      compressionSavingsBits: Math.max(0, initialBits - totalDescriptionBits),
    };
  }

  private countUncovered(rules: readonly MdlRuleItem[], facts: readonly string[]): number {
    const covered = new Set<string>();
    for (const r of rules) {
      for (const f of r.covers) covered.add(f);
    }
    let missing = 0;
    for (const f of facts) {
      if (!covered.has(f)) missing++;
    }
    return missing;
  }
}

export interface AgentBid {
  readonly agentId: string;
  readonly bid: number;
}

/**
 * 5. 게임이론 비례 배분 경매 스케줄러 (Kelly Proportional Allocation)
 * 최소 보장분(reserveFloor)으로 기아 상태(Starvation)를 방지하면서 입찰 비율대로 예산을 배분합니다.
 */
export class KellyAuctionScheduler {
  public allocate(
    bids: readonly AgentBid[],
    totalBudget: number,
    reserveFloor: number = 0
  ): Record<string, number> {
    const result: Record<string, number> = {};
    if (bids.length === 0 || totalBudget <= 0) return result;

    const safeFloor = Math.max(0, Math.min(reserveFloor, totalBudget / bids.length));
    const reservedTotal = safeFloor * bids.length;
    const auctionPool = Math.max(0, totalBudget - reservedTotal);

    const sanitizedBids = bids.map((b) => ({
      agentId: b.agentId,
      bid: Number.isFinite(b.bid) && b.bid > 0 ? b.bid : 0,
    }));
    const bidSum = sanitizedBids.reduce((acc, b) => acc + b.bid, 0);

    for (const item of sanitizedBids) {
      const proportional =
        bidSum > 0 ? auctionPool * (item.bid / bidSum) : auctionPool / sanitizedBids.length;
      result[item.agentId] = Number((safeFloor + proportional).toFixed(4));
    }

    return result;
  }
}

export interface RaadAssessment {
  readonly isAnomaly: boolean;
  readonly maxNormalSimilarity: number;
  readonly anomalyScore: number;
}

/**
 * 6. 검색 증강 이상 탐지 (RAAD: Retrieval-Augmented Anomaly Detection)
 * 정상 템플릿과 신규 로그/텔레메트리 간의 토큰 자카드 유사도를 대조해 오탐을 줄입니다.
 */
export class RaadTemplateMatcher {
  private readonly normalTemplates: Set<string>[] = [];

  constructor(
    private readonly anomalyThreshold: number = INTERDISCIPLINARY_CONFIG.DEFAULT_RAAD_ANOMALY_THRESHOLD
  ) {}

  public addNormalTemplate(text: string): void {
    this.normalTemplates.push(this.tokenize(text));
  }

  public evaluate(text: string): RaadAssessment {
    if (this.normalTemplates.length === 0) {
      return { isAnomaly: false, maxNormalSimilarity: 1, anomalyScore: 0 };
    }
    const target = this.tokenize(text);
    let maxSim = 0;

    for (const tpl of this.normalTemplates) {
      const sim = this.jaccard(target, tpl);
      if (sim > maxSim) maxSim = sim;
    }

    const anomalyScore = Number((1 - maxSim).toFixed(4));
    return {
      isAnomaly: anomalyScore >= this.anomalyThreshold,
      maxNormalSimilarity: Number(maxSim.toFixed(4)),
      anomalyScore,
    };
  }

  private tokenize(text: string): Set<string> {
    return new Set(
      text
        .toLowerCase()
        .split(/[^a-z0-9가-힣_]+/)
        .filter(Boolean)
    );
  }

  private jaccard(a: Set<string>, b: Set<string>): number {
    if (a.size === 0 && b.size === 0) return 1;
    let intersection = 0;
    for (const tok of a) {
      if (b.has(tok)) intersection++;
    }
    const union = a.size + b.size - intersection;
    return union === 0 ? 0 : intersection / union;
  }
}
