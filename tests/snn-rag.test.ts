import { describe, expect, test } from 'bun:test';
import {
  HybridSnnRagEngine,
  SNN_RAG_CONFIG,
  tokenizeForSnn,
  type NeuralRerankCandidate,
  type SnnRagDocument,
} from '../memory/snn-rag';
import {
  createHttpReranker,
  createOllamaEmbedder,
  createOpenAiCompatibleEmbedder,
} from '../memory/snn-neural-adapters';

describe('SNN-RAG 하이브리드 엔진 (Spiking Neural Network + Neural RAG)', () => {
  test('1. Event-Driven LIF 스파이크 게이팅이 대규모 코퍼스에서 신경망 연산을 대폭 절감한다', async () => {
    let embedCallCount = 0;
    const countingEmbedder = (text: string): number[] => {
      embedCallCount++;
      // 간단한 결정론적 8차원 테스트 임베딩
      const v = new Array<number>(8).fill(0);
      for (let i = 0; i < text.length; i++) {
        v[i % 8] += (text.charCodeAt(i) % 13) / 13;
      }
      return v;
    };

    const engine = new HybridSnnRagEngine({
      neuralEmbedder: countingEmbedder,
    });

    const docs: SnnRagDocument[] = [];
    for (let i = 0; i < 200; i++) {
      docs.push({
        id: `noise-${i}`,
        title: `일반 로그 항목 ${i}`,
        text: `시스템 일반 상태 보고서 번호 ${i} 정상 작동 중 네트워크 지연 없음`,
      });
    }
    docs.push(
      {
        id: 'target-wasm',
        title: 'WASM 샌드박스 격리 정책',
        text: 'Synapse WASM 정책 엔진과 MicroSandboxVM은 프로토타입 오염과 무한루프를 차단한다.',
        tags: ['wasm', 'sandbox', 'security'],
      },
      {
        id: 'target-sqlite',
        title: 'SQLite WAL 로컬 퍼스트 스토리지',
        text: 'SQLx SQLite WAL 모드는 원자적 트랜잭션과 고성능 동시 읽기를 보장한다.',
        tags: ['sqlite', 'wal', 'database'],
      }
    );
    engine.upsertBatch(docs);

    const res = await engine.search('WASM 샌드박스 프로토타입 오염 차단', {
      topK: 3,
      maxNeuralCandidates: 8,
    });

    expect(res.hits.length).toBeGreaterThan(0);
    expect(res.hits[0].id).toBe('target-wasm');
    expect(res.hits[0].spikeCount).toBeGreaterThan(0);
    // 202건 중 최대 8건 후보만 신경망 임베딩 평가 -> 95% 이상 연산 절감
    expect(res.telemetry.neuralCandidatesEvaluated).toBeLessThanOrEqual(8);
    expect(res.telemetry.neuralComputeSavedRatio).toBeGreaterThanOrEqual(0.95);
    // 쿼리 1회 + 후보(<=8회) = 최대 9회만 호출 (전체 202건 대비 극소수)
    expect(embedCallCount).toBeLessThanOrEqual(9);
  });

  test('2. 다중 홉 시냅스 공명 전파(Multi-Hop)로 키워드가 직접 없는 연관 문서까지 발화시킨다', async () => {
    const engine = new HybridSnnRagEngine({
      timeSteps: 8,
      maxHops: 2,
      hopDecay: 0.65,
      firingThreshold: 0.45,
    });

    engine.upsertBatch([
      {
        id: 'doc-auth-entry',
        title: '인증 게이트웨이 정책',
        text: '외부 요청은 Bearer 토큰 검증을 통과해야 한다.',
        links: ['doc-key-rotation'],
      },
      {
        id: 'doc-key-rotation',
        title: '암호화 키 순환 절차',
        // 쿼리 키워드("Bearer", "토큰")가 전혀 없지만 시냅스 링크로 연결됨
        text: '매월 1일 HSM 마스터 비밀키를 자동 교체하고 감사 기록을 보관한다.',
      },
      {
        id: 'doc-unrelated',
        title: '점심 메뉴 안내',
        text: '오늘 구내식당 메뉴는 김치찌개와 계란말이입니다.',
      },
    ]);

    const res = await engine.search('Bearer 토큰 인증', { topK: 5 });
    const ids = res.hits.map((h) => h.id);

    expect(ids).toContain('doc-auth-entry');
    expect(ids).toContain('doc-key-rotation');
    const propagated = res.hits.find((h) => h.id === 'doc-key-rotation');
    expect(propagated?.hopDistance).toBe(1);
    expect(ids).not.toContain('doc-unrelated');
  });

  test('3. STDP 온라인 시냅스 가소성(LTP/LTD)으로 동의어·선호 문서를 무재인덱싱 학습한다', async () => {
    const engine = new HybridSnnRagEngine();

    engine.upsertBatch([
      {
        id: 'doc-car-manual',
        title: '자동차 정비 가이드',
        text: '엔진 오일과 브레이크 패드 교체 주기를 주기적으로 점검해야 한다.',
      },
      {
        id: 'doc-car-noise',
        title: '차량 주차장 이용 규칙',
        text: '차량 출입 시 차단기 인식을 위해 천천히 진입해 주세요.',
      },
    ]);

    // 초기에는 "차량" 키워드가 직접 들어간 주차장 문서가 상단
    const before = await engine.search('차량 정비', { topK: 2 });
    expect(before.hits.length).toBeGreaterThan(0);

    // STDP 피드백: "차량 정비" 쿼리에 대해 doc-car-manual 강화(LTP), doc-car-noise 약화(LTD)
    for (let i = 0; i < 3; i++) {
      engine.recordFeedback('차량 정비', ['doc-car-manual'], ['doc-car-noise']);
    }

    expect(engine.getSynapseWeight('차량', 'doc-car-manual')).toBeGreaterThan(1.5);
    expect(engine.getSynapseWeight('차량', 'doc-car-noise')).toBeLessThan(1.0);

    const after = await engine.search('차량 정비', { topK: 2 });
    expect(after.hits[0].id).toBe('doc-car-manual');
  });

  test('4. 플러그형 Neural Reranker와 Active Inference 토큰 예산 선택이 정상 통합된다', async () => {
    let rerankerCandidatesCount = 0;
    const mockReranker = async (
      _query: string,
      candidates: readonly NeuralRerankCandidate[]
    ): Promise<Record<string, number>> => {
      rerankerCandidatesCount = candidates.length;
      const scores: Record<string, number> = {};
      for (const c of candidates) {
        scores[c.id] = c.id === 'doc-precise' ? 0.99 : 0.25;
      }
      return scores;
    };

    const engine = new HybridSnnRagEngine({
      neuralReranker: mockReranker,
      weights: { SNN: 0.2, BM25: 0.1, DENSE: 0.2, RERANK: 0.5 },
    });

    engine.upsertBatch([
      {
        id: 'doc-broad',
        title: '분산 캐시 개요',
        text: '분산 캐시 아키텍처와 메모리 관리 전략 전반에 대한 긴 설명 문서입니다.',
        tokenCost: 120,
      },
      {
        id: 'doc-precise',
        title: '분산 캐시 스탬피드 방지',
        text: '분산 캐시 스탬피드는 확률적 조기 만료(PER)와 뮤텍스 락으로 방지한다.',
        tokenCost: 45,
      },
      {
        id: 'doc-extra',
        title: '분산 캐시 모니터링 지표',
        text: '분산 캐시 적중률과 퇴출 비율을 실시간으로 추적한다.',
        tokenCost: 40,
      },
    ]);

    const res = await engine.search('분산 캐시 장애 방지', {
      topK: 3,
      tokenBudget: 100,
    });

    expect(rerankerCandidatesCount).toBe(3);
    expect(res.hits[0].id).toBe('doc-precise');
    expect(res.hits[0].rerankScore).toBe(0.99);
    expect(res.selectedByBudgetIds).toBeDefined();
    expect(res.selectedByBudgetIds).toContain('doc-precise');
    expect(res.totalSelectedTokens).toBeLessThanOrEqual(100);
  });

  test('5. 외부 신경망 임베더/리랭커 장애 시 크래시 없이 로컬 폴백으로 무중단 동작한다', async () => {
    const failingEngine = new HybridSnnRagEngine({
      neuralEmbedder: async () => {
        throw new Error('임베딩 API 타임아웃');
      },
      neuralReranker: async () => {
        throw new Error('리랭커 서버 503');
      },
    });

    failingEngine.upsert({
      id: 'doc-resilient',
      title: '장애 복구 가이드',
      text: '네트워크 장애 시 로컬 폴백 엔진으로 자동 전환되어 무중단 서비스를 제공한다.',
    });

    const res = await failingEngine.search('장애 복구 로컬 폴백', { topK: 3 });

    expect(res.hits.length).toBe(1);
    expect(res.hits[0].id).toBe('doc-resilient');
    expect(res.telemetry.fallbackTriggered).toBe(true);
    expect(res.telemetry.warningMessage).toContain('로컬 폴백');
    expect(res.telemetry.warningMessage).toContain('리랭커 호출 실패');
  });

  test('6. 문서 갱신/삭제, 토큰화, 빈 쿼리 및 잘못된 입력 검증', async () => {
    const engine = new HybridSnnRagEngine();
    expect(tokenizeForSnn('스파이킹 신경망 SNN_v1').length).toBeGreaterThan(2);

    expect(() => engine.upsert({ id: '', text: 'invalid' })).toThrow();

    engine.upsert({ id: 'd1', text: '초기 버전 Rust 컴파일러' });
    expect(engine.size()).toBe(1);

    engine.upsert({ id: 'd1', text: '갱신 버전 Svelte 리액티브 런타임' });
    expect(engine.size()).toBe(1);

    const updatedRes = await engine.search('Svelte 리액티브', { topK: 1 });
    expect(updatedRes.hits[0].id).toBe('d1');

    expect(engine.remove('d1')).toBe(true);
    expect(engine.remove('d1')).toBe(false);
    expect(engine.size()).toBe(0);

    const emptyRes = await engine.search('   ');
    expect(emptyRes.hits.length).toBe(0);
    expect(SNN_RAG_CONFIG.DEFAULT_TIME_STEPS).toBeGreaterThan(0);
  });

  test('7. 프로덕션 HTTP 어댑터(OpenAI/Ollama/TEI Rerank) 및 수상돌기 Dense 모드로 0% 어휘 중복 문서를 검색한다', async () => {
    const server = Bun.serve({
      hostname: '127.0.0.1',
      port: 0,
      async fetch(req) {
        const url = new URL(req.url);
        const body = (await req.json()) as Record<string, unknown>;

        if (url.pathname === '/v1/embeddings') {
          const inputs = (body.input as string[]) ?? [];
          const data = inputs.map((txt, index) => {
            const lower = txt.toLowerCase();
            // 의미 축 시뮬레이션: 자동차/차량/정비 -> 축0, 요리/레시피 -> 축1
            const isAuto = lower.includes('자동차') || lower.includes('차량') || lower.includes('브레이크') || lower.includes('제동');
            const isFood = lower.includes('김치') || lower.includes('요리');
            return {
              index,
              embedding: isAuto ? [0.98, 0.05, 0.1, 0.0] : isFood ? [0.02, 0.99, 0.0, 0.1] : [0.2, 0.2, 0.8, 0.2],
            };
          });
          return Response.json({ data });
        }

        if (url.pathname === '/api/embed') {
          const inputs = (body.input as string[]) ?? [];
          return Response.json({ embeddings: inputs.map(() => [0.5, 0.5, 0.5, 0.5]) });
        }

        if (url.pathname === '/rerank') {
          const docs = (body.documents as string[]) ?? [];
          const results = docs.map((d, index) => ({
            index,
            relevance_score: d.includes('자동차') ? 3.2 : -1.8, // 로짓 반환 (applySigmoid 검증)
          }));
          return Response.json({ results });
        }

        return new Response('Not Found', { status: 404 });
      },
    });

    try {
      const baseUrl = `http://127.0.0.1:${server.port}`;
      const passageEmbedder = createOpenAiCompatibleEmbedder({
        baseUrl,
        model: 'multilingual-e5-small',
        inputPrefix: 'passage: ',
      });
      const queryEmbedder = createOpenAiCompatibleEmbedder({
        baseUrl,
        model: 'multilingual-e5-small',
        inputPrefix: 'query: ',
      });
      const ollama = createOllamaEmbedder({ baseUrl, model: 'nomic-embed-text' });
      const ollamaVec = await ollama.embed('테스트');
      expect(ollamaVec.length).toBe(4);

      const reranker = createHttpReranker({
        baseUrl,
        model: 'bge-reranker-base',
        applySigmoid: true,
      });

      const engine = new HybridSnnRagEngine({
        neuralEmbedder: passageEmbedder.embed,
        queryEmbedder: queryEmbedder.embed,
        neuralReranker: reranker,
      });

      engine.upsertBatch([
        {
          id: 'doc-auto-zero-overlap',
          title: '자동차 브레이크 계통 지침',
          text: '자동차 브레이크 패드 마모 한계선 도달 시 즉시 신품으로 교환한다.',
        },
        {
          id: 'doc-cooking',
          title: '전통 김치 요리법',
          text: '배추를 소금에 절인 뒤 양념을 버무려 숙성시킨다.',
        },
      ]);

      const preCount = await engine.precomputeEmbeddings(passageEmbedder.embedBatch);
      expect(preCount).toBe(2);

      // 쿼리("차량 제동")와 정답 문서("자동차 브레이크...")는 겹치는 단어가 0개지만 수상돌기 전류로 발화!
      const res = await engine.search('차량 제동', { topK: 2 });
      expect(res.hits.length).toBeGreaterThan(0);
      expect(res.hits[0].id).toBe('doc-auto-zero-overlap');
      expect(res.hits[0].rerankScore).toBeGreaterThan(0.9);
      expect(res.telemetry.fallbackTriggered).toBe(false);
    } finally {
      server.stop(true);
    }
  });
});
