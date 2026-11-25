/**
 * 시냅스 옵틱 적응형 페이지 정착 감지기 (Adaptive Settle & Fail-Fast Idle)
 * - 고정된 500ms~3000ms 슬립 대기 대신 DOM 변이 및 진행 중 요청이 정착(quiet window)되면 즉시 반환
 * - 페이지가 완전 유휴 상태(요청 0건, 타이머 0건, 변이 없음)인데 조건이 충족되지 않으면 타임아웃 전 조기 실패(Fail-Fast)
 */
import { OPTIC_FLASH_DEFAULTS } from "./config";

export interface PageActivitySnapshot {
  readonly inFlightRequests: number;
  readonly mutationCount: number;
  readonly pendingTimers?: number;
}

export interface AdaptiveSettleOptions {
  readonly quietWindowMs?: number;
  readonly failFastIdleMs?: number;
  readonly maxWaitMs?: number;
  readonly pollIntervalMs?: number;
  readonly failFastWhenIdle?: boolean;
}

export type SettleStatus = "pending" | "settled" | "fail_fast_idle" | "timeout";

export interface AdaptiveWaitResult {
  readonly ok: boolean;
  readonly status: SettleStatus;
  readonly elapsedMs: number;
  readonly observedMutations: number;
}

export class AdaptiveSettleTracker {
  private readonly quietWindowMs: number;
  private readonly failFastIdleMs: number;
  private readonly maxWaitMs: number;
  private readonly failFastWhenIdle: boolean;

  private startMs: number;
  private lastActivityMs: number;
  private lastMutationCount = -1;
  private totalObservedMutations = 0;

  constructor(options: AdaptiveSettleOptions = {}, nowMs: number = performance.now()) {
    this.quietWindowMs = Math.max(10, options.quietWindowMs ?? OPTIC_FLASH_DEFAULTS.ADAPTIVE_QUIET_WINDOW_MS);
    this.failFastIdleMs = Math.max(
      this.quietWindowMs,
      options.failFastIdleMs ?? OPTIC_FLASH_DEFAULTS.FAIL_FAST_IDLE_MS
    );
    this.maxWaitMs = Math.max(50, options.maxWaitMs ?? OPTIC_FLASH_DEFAULTS.ADAPTIVE_MAX_WAIT_MS);
    this.failFastWhenIdle = options.failFastWhenIdle ?? true;
    this.startMs = nowMs;
    this.lastActivityMs = nowMs;
  }

  public reset(nowMs: number = performance.now()): void {
    this.startMs = nowMs;
    this.lastActivityMs = nowMs;
    this.lastMutationCount = -1;
    this.totalObservedMutations = 0;
  }

  /**
   * 현재 활동 스냅샷과 목표 조건 충족 여부를 바탕으로 정착/조기실패/대기 판정
   */
  public step(
    activity: PageActivitySnapshot,
    conditionMet: boolean = true,
    nowMs: number = performance.now()
  ): SettleStatus {
    if (this.lastMutationCount === -1) {
      this.lastMutationCount = activity.mutationCount;
    } else if (activity.mutationCount !== this.lastMutationCount) {
      const delta = Math.max(0, activity.mutationCount - this.lastMutationCount);
      this.totalObservedMutations += delta;
      this.lastMutationCount = activity.mutationCount;
      this.lastActivityMs = nowMs;
    }

    if (activity.inFlightRequests > 0) {
      this.lastActivityMs = nowMs;
    }

    const elapsed = nowMs - this.startMs;
    const quietDuration = nowMs - this.lastActivityMs;

    if (conditionMet && activity.inFlightRequests === 0 && quietDuration >= this.quietWindowMs) {
      return "settled";
    }

    if (
      !conditionMet &&
      this.failFastWhenIdle &&
      activity.inFlightRequests === 0 &&
      (activity.pendingTimers ?? 0) === 0 &&
      quietDuration >= this.failFastIdleMs
    ) {
      return "fail_fast_idle";
    }

    if (elapsed >= this.maxWaitMs) {
      return "timeout";
    }

    return "pending";
  }

  public getObservedMutations(): number {
    return this.totalObservedMutations;
  }
}

/**
 * 비동기 조건 및 페이지 활동 콜백 기반 적응형 정착 대기 실행기
 */
export async function waitForAdaptiveSettle(
  getActivity: () => PageActivitySnapshot | Promise<PageActivitySnapshot>,
  checkCondition: () => boolean | Promise<boolean> = () => true,
  options: AdaptiveSettleOptions = {}
): Promise<AdaptiveWaitResult> {
  const start = performance.now();
  const tracker = new AdaptiveSettleTracker(options, start);
  const pollMs = Math.max(5, options.pollIntervalMs ?? 15);

  while (true) {
    const [activity, matched] = await Promise.all([getActivity(), checkCondition()]);
    const now = performance.now();
    const status = tracker.step(activity, matched, now);

    if (status === "settled") {
      return {
        ok: true,
        status,
        elapsedMs: Number((now - start).toFixed(2)),
        observedMutations: tracker.getObservedMutations(),
      };
    }

    if (status === "fail_fast_idle" || status === "timeout") {
      return {
        ok: false,
        status,
        elapsedMs: Number((now - start).toFixed(2)),
        observedMutations: tracker.getObservedMutations(),
      };
    }

    await new Promise((r) => setTimeout(r, pollMs));
  }
}
