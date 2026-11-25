import { describe, expect, it } from "bun:test";
import { chmodSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  A11yParser,
  AdaptiveSettleTracker,
  LocatorDiagnostics,
  OpticPersistentSession,
  SynapseOptic,
  waitForAdaptiveSettle,
  type OpticSnapshot,
} from "../src";

function createSampleSnapshot(): OpticSnapshot {
  const nodes = A11yParser.formatReferenceSnapshot([
    ["cafebabe1234:1", "button", "Log out"],
    ["cafebabe1234:2", "button", "로그인"],
    ["cafebabe1234:3", "textbox", "이메일 주소"],
    ["cafebabe1234:4", "link", "비밀번호 재설정"],
  ]);
  return {
    generation: "cafebabe1234",
    totalNodes: nodes.length,
    formattedPrompt: A11yParser.formatSnapshotSummary(nodes),
    nodes,
    durationMs: 1.2,
  };
}

describe("⚡ Synapse Optic Flash 업그레이드 (영속 세션 · 로케이터 진단 · 적응형 정착)", () => {
  it("1. [LocatorDiagnostics] 오타·유사 라벨(영문/한국어) 요청 시 가장 가까운 후보를 추천한다", () => {
    const snap = createSampleSnapshot();

    const enDiag = LocatorDiagnostics.diagnose("Logout", snap);
    expect(enDiag.suggestions.length).toBeGreaterThan(0);
    expect(enDiag.suggestions[0].label).toBe("Log out");
    expect(enDiag.suggestions[0].shortAlias).toBe("@1");
    expect(enDiag.formattedMessage).toContain("Log out");

    const koDiag = LocatorDiagnostics.diagnose("button:로그인하기", snap);
    expect(koDiag.suggestions.length).toBeGreaterThan(0);
    expect(koDiag.suggestions[0].label).toBe("로그인");
    expect(koDiag.suggestions[0].reference).toBe("@cafebabe1234:2");
  });

  it("2. [LocatorDiagnostics] 범위 초과 별칭(@99), 만료 세대 참조, 단축 별칭(@1) 자동 해석을 처리한다", () => {
    const snap = createSampleSnapshot();

    // 단축 별칭 @2 -> @cafebabe1234:2 정식 참조 변환
    expect(LocatorDiagnostics.resolveTargetReference("@2", snap)).toBe("@cafebabe1234:2");
    expect(LocatorDiagnostics.resolveTargetReference("button:로그인", snap)).toBe("@cafebabe1234:2");

    // 범위 초과 별칭 진단
    const outOfRange = LocatorDiagnostics.diagnose("@99", snap);
    expect(outOfRange.hint).toContain("유효 범위: @1 ~ @4");

    // 만료된 이전 세대 참조 진단 및 동일 순번 후보 제시
    const stale = LocatorDiagnostics.diagnose("@deadbeef0000:3", snap);
    expect(stale.hint).toContain("만료된 이전 스냅샷 세대");
    expect(stale.suggestions[0].label).toBe("이메일 주소");
  });

  it("3. [OpticPersistentSession] ANSI 프롬프트 REPL 자식 프로세스와 상시 파이프로 직렬 통신한다", async () => {
    const tmpDir = mkdtempSync(join(tmpdir(), "optic-mock-repl-"));
    const mockScript = join(tmpDir, "mock-prom.js");
    const launcher = process.platform === "win32" ? join(tmpDir, "mock-prom.cmd") : join(tmpDir, "mock-prom.sh");

    writeFileSync(
      mockScript,
      `
      const readline = require("node:readline");
      process.stdout.write("\\x1b[1;32m🤖 [모의 프로메테우스] 상시 연결 완료\\x1b[0m\\n");
      process.stdout.write("\\x1b[1;36mprom>\\x1b[0m ");
      const rl = readline.createInterface({ input: process.stdin, terminal: false });
      rl.on("line", (line) => {
        const t = line.trim();
        if (t === "exit") {
          process.stdout.write("세션 종료\\n");
          process.exit(0);
        } else if (t.startsWith("nav ")) {
          process.stdout.write("  이동 중...\\n  [@abcd1234:1] [button] 확인\\n  [@abcd1234:2] [textbox] 검색어\\n [완료]\\n");
        } else if (t.startsWith("click @abcd1234:1")) {
          process.stdout.write("  [성공] 클릭 완료\\n");
        } else {
          process.stdout.write("  [실패] 요소를 찾을 수 없습니다\\n");
        }
        process.stdout.write("\\x1b[1;36mprom>\\x1b[0m ");
      });
      `,
      "utf-8"
    );

    if (process.platform === "win32") {
      writeFileSync(launcher, `@echo off\r\n"${process.execPath}" "${mockScript}" %*\r\n`, "utf-8");
    } else {
      writeFileSync(launcher, `#!/bin/sh\nexec "${process.execPath}" "${mockScript}" "$@"\n`, "utf-8");
      chmodSync(launcher, 0o755);
    }

    const session = new OpticPersistentSession({
      binaryPath: launcher,
      endpoint: "http://127.0.0.1:9222",
      connectTimeoutMs: 5_000,
      commandTimeoutMs: 5_000,
    });

    try {
      await session.connect();
      expect(session.isConnected()).toBe(true);

      // 동시 3개 명령 발행 시에도 내부 직렬화 큐로 순서와 응답이 섞이지 않음
      const [navRes, okClick, failClick] = await Promise.all([
        session.executeCommand("nav https://example.com"),
        session.executeCommand("click @abcd1234:1"),
        session.executeCommand("click @missing"),
      ]);

      expect(navRes.ok).toBe(true);
      expect(navRes.output).toContain("[@abcd1234:1] [button] 확인");
      expect(okClick.ok).toBe(true);
      expect(failClick.ok).toBe(false);
      expect(failClick.error).toContain("[실패]");
    } finally {
      session.close();
      rmSync(tmpDir, { recursive: true, force: true });
    }
  });

  it("4. [AdaptiveSettle] 고정 슬립 없이 정착 즉시 반환하고 완전 유휴 시 Fail-Fast 조기 실패한다", async () => {
    const tracker = new AdaptiveSettleTracker({ quietWindowMs: 50, failFastIdleMs: 100, maxWaitMs: 1000 }, 0);

    // t=0..30ms 요청 진행 중 -> pending
    expect(tracker.step({ inFlightRequests: 1, mutationCount: 2 }, true, 30)).toBe("pending");
    // t=40ms 요청 종료 직후 -> 아직 quietWindow(50ms) 미달 -> pending
    expect(tracker.step({ inFlightRequests: 0, mutationCount: 2 }, true, 60)).toBe("pending");
    // t=95ms 마지막 활동(30ms)으로부터 65ms 경과 -> 즉시 settled!
    expect(tracker.step({ inFlightRequests: 0, mutationCount: 2 }, true, 95)).toBe("settled");

    // 비동기 Fail-Fast Idle 검증: 조건 미충족 + 유휴 상태면 maxWaitMs(2000ms) 전에 즉시 종료
    const res = await waitForAdaptiveSettle(
      () => ({ inFlightRequests: 0, mutationCount: 0, pendingTimers: 0 }),
      () => false,
      { quietWindowMs: 20, failFastIdleMs: 45, maxWaitMs: 2_000, pollIntervalMs: 10 }
    );
    expect(res.ok).toBe(false);
    expect(res.status).toBe("fail_fast_idle");
    expect(res.elapsedMs).toBeLessThan(400);
  });

  it("5. [SynapseOptic 통합] 액션 실패 시 활성 스냅샷 기반 유사 후보가 에러 결과에 자동 첨부된다", async () => {
    const optic = new SynapseOptic({
      endpoint: "http://127.0.0.1:59998",
      timeoutMs: 2_000,
    });
    optic.setActiveSnapshot(createSampleSnapshot());

    const res = await optic.click("Logout");
    expect(res.ok).toBe(false);
    expect(res.suggestions).toBeDefined();
    expect(res.suggestions![0].label).toBe("Log out");
    expect(res.error).toContain("유사 요소 후보");
    expect(res.error).toContain("Log out");
  });
});
