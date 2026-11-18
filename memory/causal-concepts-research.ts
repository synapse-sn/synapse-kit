import {
  ActiveInferenceContextSelector,
  HomeostaticController,
  ImmuneAnomalyDetector,
  KellyAuctionScheduler,
  MdlGraphDistiller,
  RaadTemplateMatcher,
  type ContextCandidate,
} from './causal-concepts';

function jainsFairness(values: number[]): number {
  const n = values.length;
  if (n === 0) return 1;
  const sum = values.reduce((a, b) => a + b, 0);
  const sqSum = values.reduce((a, b) => a + b * b, 0);
  return sqSum === 0 ? 1 : (sum * sum) / (n * sqSum);
}

// 1. Active Inference vs. Naive Relevance Top-K (500개 후보, 300 토큰 예산)
const candidates: ContextCandidate[] = Array.from({ length: 500 }, (_, i) => {
  const isCompactHighInfo = i % 10 === 0;
  return {
    id: `rule-${i}`,
    tokenCost: isCompactHighInfo ? 25 + (i % 15) : 120 + (i % 180),
    relevance: isCompactHighInfo ? 0.88 : 0.92,
    epistemicGain: isCompactHighInfo ? 0.95 : 0.35,
  };
});

const budget = 300;
const naiveSorted = [...candidates].sort((a, b) => b.relevance - a.relevance);
let naiveTokens = 0;
let naiveEpistemic = 0;
let naiveCount = 0;
for (const c of naiveSorted) {
  if (naiveTokens + c.tokenCost <= budget) {
    naiveTokens += c.tokenCost;
    naiveEpistemic += c.epistemicGain;
    naiveCount++;
  }
}

const t0 = performance.now();
const aif = new ActiveInferenceContextSelector(0.002);
const aifRes = aif.select(candidates, budget);
const aifMs = performance.now() - t0;
const aifEpistemic = aifRes.selectedIds.reduce((acc, id) => {
  const found = candidates.find((c) => c.id === id);
  return acc + (found?.epistemicGain ?? 0);
}, 0);

// 2. Homeostatic PID vs. Static Forgetting (50틱 부하 스파이크 시뮬레이션)
const pid = new HomeostaticController(50);
let pidLoad = 50;
let staticLoad = 50;
const pidTrajectory: number[] = [];
const staticTrajectory: number[] = [];
for (let tick = 0; tick < 50; tick++) {
  const burst = tick >= 10 && tick < 25 ? 25 : 0;
  const pidOut = pid.step(pidLoad, 1);
  pidLoad = Math.max(20, pidLoad + burst - pidOut.recommendedForgettingRate * pidLoad * 0.6);
  staticLoad = Math.max(20, staticLoad + burst - 0.05 * staticLoad * 0.6);
  pidTrajectory.push(pidLoad);
  staticTrajectory.push(staticLoad);
}
const pidPeak = Math.max(...pidTrajectory);
const staticPeak = Math.max(...staticTrajectory);
const pidFinal = pidTrajectory[pidTrajectory.length - 1];
const staticFinal = staticTrajectory[staticTrajectory.length - 1];

// 3. Immune Negative Selection & Clonal Memory (정상 200건 vs 이상 50건)
const immune = new ImmuneAnomalyDetector(0.25);
for (let i = 0; i < 200; i++) {
  immune.trainSelf([0.1 + (i % 5) * 0.01, 0.2 + (i % 3) * 0.01, 0.15]);
}
let fp = 0;
for (let i = 0; i < 100; i++) {
  if (immune.evaluate([0.12, 0.21, 0.15]).isAnomaly) fp++;
}
let tp = 0;
const tImm0 = performance.now();
for (let i = 0; i < 50; i++) {
  const vec = [0.8 + (i % 5) * 0.03, 0.85, 0.9];
  const ev = immune.evaluate(vec);
  if (ev.isAnomaly) tp++;
  if (i === 0) immune.cloneThreat('clone-attack-0', vec);
}
const immMs = performance.now() - tImm0;
const cloneHit = immune.evaluate([0.8, 0.85, 0.9]).matchedCloneId !== null;

// 4. MDL Graph Distillation (60개 중복 규칙 → 최소 커버 집합)
const facts = Array.from({ length: 30 }, (_, i) => `fact-${i}`);
const mdlRules = [
  { id: 'core-0', covers: facts.slice(0, 10) },
  { id: 'core-1', covers: facts.slice(10, 20) },
  { id: 'core-2', covers: facts.slice(20, 30) },
  ...Array.from({ length: 57 }, (_, i) => ({
    id: `dup-${i}`,
    covers: facts.slice(i % 25, (i % 25) + 3),
  })),
];
const mdl = new MdlGraphDistiller(32, 16);
const mdlRes = mdl.distill(mdlRules, facts);

// 5. Kelly Auction vs. Winner-Take-All (4개 에이전트, 총 1000 토큰)
const kelly = new KellyAuctionScheduler();
const bids = [
  { agentId: 'heavy', bid: 90 },
  { agentId: 'mid', bid: 8 },
  { agentId: 'low', bid: 2 },
  { agentId: 'idle', bid: 0 },
];
const kellyAlloc = kelly.allocate(bids, 1000, 100);
const kellyVals = Object.values(kellyAlloc);
const wtaVals = [1000, 0, 0, 0];

// 6. RAAD Template Matcher (정상 50건 + 변조 50건)
const raad = new RaadTemplateMatcher(0.65);
raad.addNormalTemplate('synapse compiler incremental check completed 0 errors in 12ms');
raad.addNormalTemplate('synapse memory resonance matched 4 rules 94% savings');
raad.addNormalTemplate('pm2 chzzk-clip-bot online memory 41mb cpu 0%');

let raadFp = 0;
let raadTp = 0;
const tRaad0 = performance.now();
for (let i = 0; i < 50; i++) {
  if (raad.evaluate(`synapse compiler incremental check completed 0 errors in ${10 + i}ms`).isAnomaly) raadFp++;
  if (raad.evaluate(`panic sigsegv unauthorized /bin/sh exec payload ${i}`).isAnomaly) raadTp++;
}
const raadMs = performance.now() - tRaad0;

process.stdout.write(
  JSON.stringify(
    {
      activeInference: {
        naive: { count: naiveCount, tokens: naiveTokens, epistemicSum: Number(naiveEpistemic.toFixed(2)) },
        aif: { count: aifRes.selectedIds.length, tokens: aifRes.totalTokens, epistemicSum: Number(aifEpistemic.toFixed(2)), latencyMs: Number(aifMs.toFixed(3)) },
      },
      homeostaticPid: {
        staticPeak: Number(staticPeak.toFixed(2)),
        staticFinal: Number(staticFinal.toFixed(2)),
        pidPeak: Number(pidPeak.toFixed(2)),
        pidFinal: Number(pidFinal.toFixed(2)),
      },
      immune: {
        fpRate: fp / 100,
        tpRate: tp / 50,
        cloneHit,
        eval50Ms: Number(immMs.toFixed(3)),
      },
      mdl: {
        inputRules: mdlRules.length,
        retainedRules: mdlRes.retainedIds.length,
        pruneRatioPct: Number((((mdlRules.length - mdlRes.retainedIds.length) / mdlRules.length) * 100).toFixed(1)),
        savingsBits: mdlRes.compressionSavingsBits,
      },
      kellyAuction: {
        kellyFairness: Number(jainsFairness(kellyVals).toFixed(4)),
        wtaFairness: Number(jainsFairness(wtaVals).toFixed(4)),
        starvedAgentsKelly: kellyVals.filter((v) => v <= 0).length,
        starvedAgentsWta: wtaVals.filter((v) => v <= 0).length,
      },
      raad: {
        fpRate: raadFp / 50,
        tpRate: raadTp / 50,
        eval100Ms: Number(raadMs.toFixed(3)),
      },
    },
    null,
    2
  ) + '\n'
);
