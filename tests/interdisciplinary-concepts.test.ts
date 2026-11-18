import { describe, expect, it } from 'bun:test';
import {
  ActiveInferenceContextSelector,
  HomeostaticController,
  ImmuneAnomalyDetector,
  KellyAuctionScheduler,
  MdlGraphDistiller,
  RaadTemplateMatcher,
} from '../memory/causal-concepts';

describe('🧪 시냅스 초학제적 6대 개념 구현 및 정량 연구 검증', () => {
  it('1. [능동 추론 / FEP] 제한된 토큰 예산 내에서 자유 에너지 감소 밀도가 높은 컨텍스트만 선택한다', () => {
    const selector = new ActiveInferenceContextSelector(0.002);
    const result = selector.select(
      [
        { id: 'bloated-doc', tokenCost: 500, relevance: 0.9, epistemicGain: 0.9 }, // net < 0 or low density
        { id: 'core-invariant', tokenCost: 40, relevance: 0.95, epistemicGain: 0.95 },
        { id: 'compact-rule', tokenCost: 30, relevance: 0.85, epistemicGain: 0.9 },
        { id: 'noise', tokenCost: 80, relevance: 0.05, epistemicGain: 0.05 },
      ],
      100
    );

    expect(result.selectedIds).toEqual(['compact-rule', 'core-invariant']);
    expect(result.totalTokens).toBe(70);
    expect(result.totalFreeEnergyReduction).toBeGreaterThan(1.4);
  });

  it('2. [항상성 PID 제어] 부하 급증 시 망각률(forgettingRate)을 높이고 안정화 시 낮춘다', () => {
    const pid = new HomeostaticController(50); // 목표 지연 50ms
    const overloaded = pid.step(120, 1);
    expect(overloaded.error).toBe(70);
    expect(overloaded.controlSignal).toBeGreaterThan(0.5);
    expect(overloaded.recommendedForgettingRate).toBeGreaterThan(0.25);

    pid.reset();
    const relaxed = pid.step(20, 1);
    expect(relaxed.controlSignal).toBeLessThan(0);
    expect(relaxed.recommendedForgettingRate).toBe(0.05);
  });

  it('3. [인공 면역 체계] Self 공간 외부의 Non-Self 이상을 탐지하고 클론 메모리에 각인한다', () => {
    const immune = new ImmuneAnomalyDetector(0.2);
    immune.trainSelf([0.1, 0.1, 0.1]);
    immune.trainSelf([0.15, 0.12, 0.08]);

    const normalEval = immune.evaluate([0.12, 0.11, 0.09]);
    expect(normalEval.isAnomaly).toBe(false);

    const unknownThreat = [0.9, 0.85, 0.92];
    const threatEval = immune.evaluate(unknownThreat);
    expect(threatEval.isAnomaly).toBe(true);
    expect(threatEval.matchedCloneId).toBeNull();

    immune.cloneThreat('threat-0x1', unknownThreat);
    const recalledEval = immune.evaluate([0.89, 0.86, 0.91]);
    expect(recalledEval.isAnomaly).toBe(true);
    expect(recalledEval.matchedCloneId).toBe('threat-0x1');
  });

  it('4. [MDL 그래프 증류] 중복 규칙을 제거해 최소 설명 길이(Minimum Description Length)로 압축한다', () => {
    const distiller = new MdlGraphDistiller(32, 16);
    const facts = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6'];
    const result = distiller.distill(
      [
        { id: 'r-broad-1', covers: ['f1', 'f2', 'f3', 'f4'] }, // gain = 64 - 32 = 32
        { id: 'r-redundant', covers: ['f1', 'f2'] }, // 중복
        { id: 'r-broad-2', covers: ['f4', 'f5', 'f6'] }, // 나머지 f5,f6 커버 (32-32=0 -> 제외 또는 임계 조정 확인)
        { id: 'r-tail', covers: ['f4', 'f5', 'f6', 'f1'] },
      ],
      facts
    );

    expect(result.retainedIds).toContain('r-broad-1');
    expect(result.retainedIds).not.toContain('r-redundant');
    expect(result.compressionSavingsBits).toBeGreaterThan(0);
  });

  it('5. [게임이론 Kelly 경매] 최소 보장분으로 기아를 방지하면서 입찰 비율대로 예산을 배분한다', () => {
    const scheduler = new KellyAuctionScheduler();
    const alloc = scheduler.allocate(
      [
        { agentId: 'agent-heavy', bid: 75 },
        { agentId: 'agent-light', bid: 25 },
        { agentId: 'agent-zero', bid: 0 },
      ],
      1000,
      100
    );

    expect(alloc['agent-zero']).toBe(100);
    expect(alloc['agent-heavy']).toBe(100 + 700 * 0.75);
    expect(alloc['agent-light']).toBe(100 + 700 * 0.25);
    const sum = alloc['agent-heavy'] + alloc['agent-light'] + alloc['agent-zero'];
    expect(sum).toBe(1000);
  });

  it('6. [RAAD 이상 탐지] 정상 로그 템플릿과 비교해 변조/비정상 텔레메트리만 분리 탐지한다', () => {
    const raad = new RaadTemplateMatcher(0.65);
    raad.addNormalTemplate('synapse compiler incremental check completed 0 errors in 12ms');
    raad.addNormalTemplate('synapse memory resonance matched 3 rules 94% savings');

    const normal = raad.evaluate('synapse compiler incremental check completed 0 errors in 18ms');
    expect(normal.isAnomaly).toBe(false);
    expect(normal.maxNormalSimilarity).toBeGreaterThan(0.7);

    const weird = raad.evaluate('segfault panic core dumped unauthorized shell exec /bin/sh');
    expect(weird.isAnomaly).toBe(true);
    expect(weird.anomalyScore).toBeGreaterThanOrEqual(0.65);
  });
});

import { RingBufferBus } from '../bus/src/ringBufferBus';
import { SnEngine, attachCognitiveBridge } from '../sn/src';

describe('🔗 sn 제어 플레인 ↔ 초학제적 인지·면역 엔진 결합 검증', () => {
  it('7. [sn + Immune + Bus] 정상(Self) 페이로드는 통과시키고 미지(Non-Self) 이상 페이로드는 dead_letter로 차단한다', async () => {
    const script = `
      native fn sn_immune_gate(x: i32, y: i32, z: i32) -> i32 via ffi "sn_cognitive";

      invariant immune_self_ok {
        cause "인공 면역 Self 경계 이탈 탐지";
        require sn_immune_gate(payload.x, payload.y, payload.z) == 1;
        severity critical;
      }

      wire cognitive_guard_wire {
        from "cog.in";
        check immune_self_ok;
        to "cog.out";
        on_error "cog.dead_letter";
      }
    `;

    const bus = new RingBufferBus({ ringCapacity: 32 });
    const engine = new SnEngine(script);
    await engine.init();

    const immune = new ImmuneAnomalyDetector(0.25);
    immune.trainSelf([0.1, 0.2, 0.15]);
    attachCognitiveBridge(engine, { immune, scaleFactor: 1000 });

    const passed: any[] = [];
    const blocked: any[] = [];
    bus.subscribe('cog.out', (m) => passed.push(m.payload));
    bus.subscribe('cog.dead_letter', (m) => blocked.push(m.payload));

    engine.bindBus(bus);
    bus.emit('cog.in', { x: 120, y: 210, z: 150 }); // Self -> 통과
    bus.emit('cog.in', { x: 900, y: 880, z: 950 }); // Non-Self -> 차단

    expect(passed.length).toBe(1);
    expect(passed[0].x).toBe(120);
    expect(blocked.length).toBe(1);
    expect(blocked[0].payload.x).toBe(900);

    engine.unbindBus();
  });
});
