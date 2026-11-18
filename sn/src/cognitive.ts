/**
 * sn(Synapse Control Language) ↔ 초학제적 인지·제어 엔진 통합 브리지
 * - .sn 불변식(invariant)과 와이어(wire)에서 인공 면역 탐지(ImmuneAnomalyDetector) 및
 *   항상성 PID 제어(HomeostaticController)를 네이티브 게이트 함수로 직접 호출할 수 있게 결합합니다.
 */
import type { SnEngine } from "./engine";

export interface ImmuneDetectorLike {
  evaluate(vector: readonly number[]): { readonly isAnomaly: boolean };
}

export interface PidControllerLike {
  step(measured: number, dt?: number): { readonly recommendedForgettingRate: number };
}

export interface CognitiveBridgeProviders {
  readonly immune?: ImmuneDetectorLike;
  readonly pid?: PidControllerLike;
  readonly scaleFactor?: number;
}

/**
 * SnEngine에 초학제적 인지/면역/PID FFI 게이트 핸들러를 등록합니다.
 * - sn_immune_gate(x, y, z): 정상(Self)이면 1, 이상(Non-Self)이면 0 반환 (scaleFactor 기준 정수화)
 * - sn_pid_forgetting(measured): 현재 부하에 따른 권장 망각률(0..1000) 반환
 */
export function attachCognitiveBridge(
  engine: SnEngine,
  providers: CognitiveBridgeProviders = {},
): void {
  const scale = providers.scaleFactor ?? 1000;

  if (providers.immune) {
    const immune = providers.immune;
    engine.registerFfiHandler("sn_immune_gate", (x: number, y: number, z: number) => {
      const res = immune.evaluate([x / scale, y / scale, z / scale]);
      return res.isAnomaly ? 0 : 1;
    });
  }

  if (providers.pid) {
    const pid = providers.pid;
    engine.registerFfiHandler("sn_pid_forgetting", (measured: number) => {
      const out = pid.step(measured, 1);
      return Math.round(out.recommendedForgettingRate * scale);
    });
  }
}
