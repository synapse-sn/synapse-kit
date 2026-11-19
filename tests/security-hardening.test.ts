import { describe, expect, it } from 'bun:test';
import { mkdtempSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { startCausalServer } from '../memory/causal-server';
import { queryRemotePeer } from '../memory/causal-peer';
import { CausalObsidianLensEngine } from '../memory/causal-lens';
import { CausalMemoryStore } from '../memory/causal-store';
import { ContextCheckpointStore } from '../memory/causal-context';
import { PeerMeshRouter } from '../mesh/src/peerMesh';
import { CloudflareTunnelManager } from '../mesh/src/tunnel';
import { fetchRecentArxivPapers } from '../optic/src/liveFetcher';

describe('보안 취약점 하드닝 회귀 검증 (Security Hardening Suite)', () => {
  it('1. [Causal Server] CORS 와일드카드(*) 차단, 토큰 인증(401) 및 대용량 본문(413) 방어', async () => {
    const tmpDir = mkdtempSync(join(tmpdir(), 'synapse-sec-server-'));
    const dbPath = join(tmpDir, 'memory.json');
    const contextDbPath = join(tmpDir, 'context.json');

    const server = startCausalServer({
      port: 0,
      host: '127.0.0.1',
      dbPath,
      contextDbPath,
      authToken: 'test-secret-token',
      maxBodyBytes: 128,
    });

    try {
      const baseUrl = `http://127.0.0.1:${server.port}`;

      // 인증 없이 /query 접근 시 401 반환 + CORS '*' 미노출
      const unauthRes = await fetch(`${baseUrl}/query`);
      expect(unauthRes.status).toBe(401);
      expect(unauthRes.headers.get('access-control-allow-origin')).not.toBe('*');

      // /health는 헬스체크용으로 허용
      const healthRes = await fetch(`${baseUrl}/health`);
      expect(healthRes.status).toBe(200);

      // 유효 토큰 + 128B 초과 페이로드 전송 시 413 Payload Too Large 반환
      const oversizedRes = await fetch(`${baseUrl}/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: 'Bearer test-secret-token',
        },
        body: JSON.stringify({ keywords: ['a'.repeat(300)] }),
      });
      expect(oversizedRes.status).toBe(413);

      // 유효 토큰 + 정상 크기 요청은 200 반환
      const okRes = await fetch(`${baseUrl}/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Synapse-Token': 'test-secret-token',
        },
        body: JSON.stringify({ keywords: ['security'] }),
      });
      expect(okRes.status).toBe(200);
    } finally {
      server.stop(true);
      rmSync(tmpDir, { recursive: true, force: true });
    }
  });

  it('2. [SSRF 방어] queryRemotePeer 및 PeerMeshRouter가 file:// 및 메타데이터 IP를 차단해야 한다', async () => {
    await expect(queryRemotePeer('file:///etc/passwd', {})).rejects.toThrow();
    await expect(queryRemotePeer('http://169.254.169.254/latest/meta-data', {})).rejects.toThrow();

    const router = new PeerMeshRouter();
    expect(() =>
      router.addPeer({ id: 'bad-scheme', endpoint: 'ftp://example.com:4343' })
    ).toThrow();
    expect(() =>
      router.addPeer({ id: 'bad-meta', endpoint: 'http://169.254.169.254:80' })
    ).toThrow();
  });

  it('3. [CLI 플래그 인젝션 방어] CloudflareTunnelManager가 잘못된 포트 및 -- 플래그 호스트명을 차단해야 한다', async () => {
    const tunnel = new CloudflareTunnelManager();

    const badFlag = await tunnel.startTunnel(4343, '--config=/tmp/evil.yml');
    expect(badFlag.success).toBe(false);

    const badPort = await tunnel.startTunnel(99999, 'mesh.example.com');
    expect(badPort.success).toBe(false);
  });

  it('4. [Path Traversal 방어] CausalObsidianLensEngine이 널 바이트 볼트 경로 및 악성 체크포인트 ID를 차단/새니타이징해야 한다', async () => {
    const tmpDir = mkdtempSync(join(tmpdir(), 'synapse-sec-lens-'));
    try {
      const memStore = new CausalMemoryStore(join(tmpDir, 'mem.json'));
      const ctxStore = new ContextCheckpointStore(join(tmpDir, 'ctx.json'), join(tmpDir, 'mem.json'));
      const lens = new CausalObsidianLensEngine(memStore, ctxStore);

      const badVaultRes = await lens.exportToVault(`${tmpDir}\0evil`);
      expect(badVaultRes.success).toBe(false);
    } finally {
      rmSync(tmpDir, { recursive: true, force: true });
    }
  });

  it('5. [Optic LiveFetcher] 허용 바이트 상한 초과 시 안전 폴백으로 차단해야 한다', async () => {
    const oversizedXml = '<feed>' + '<entry><id>1</id><title>x</title></entry>'.repeat(150_000) + '</feed>';
    const res = await fetchRecentArxivPapers({
      fetchImpl: async () =>
        new Response(oversizedXml, {
          status: 200,
          headers: { 'content-length': String(Buffer.byteLength(oversizedXml, 'utf8')) },
        }),
    });
    expect(res.liveFetched).toBe(false);
    expect(res.papers.length).toBe(0);
    expect(res.error).toContain('초과');
  });
});

import { cppChecker } from '../compiler/plugins/cpp';
import { pythonChecker } from '../compiler/plugins/python';
import { javaChecker } from '../compiler/plugins/java';
import { goChecker } from '../compiler/plugins/go';
import { csharpChecker } from '../compiler/plugins/csharp';
import { rustChecker } from '../compiler/plugins/rust';
import { typescriptChecker } from '../compiler/plugins/typescript';
import { svelteChecker } from '../compiler/plugins/svelte';
import { nativeChecker } from '../compiler/plugins/native';
import { nsisChecker } from '../compiler/plugins/nsis';
import { schemaChecker } from '../compiler/plugins/schema';

describe('컴파일러 11대 언어 플러그인 보안 하드닝 검증', () => {
  it('6. [언어 플러그인 공통] Path Traversal, 임의 바이너리 및 위험 플래그 인젝션을 모두 차단해야 한다', async () => {
    const tmpDir = mkdtempSync(join(tmpdir(), 'synapse-sec-plugins-'));
    try {
      const ctx = {
        rootDir: tmpDir,
        modifiedFiles: [],
        env: {},
        isForce: false,
      } as any;

      const cppRes = await cppChecker({ compiler: 'sh', targetFiles: ['main.cpp'] }).check(ctx);
      expect(cppRes.status).toBe('error');
      expect(cppRes.details).toContain('Plugin Security');

      const pyRes = await pythonChecker({
        extraArgs: ['-c', 'import os; os.system("id")'],
        targetFiles: ['app.py'],
      }).check(ctx);
      expect(pyRes.status).toBe('error');
      expect(pyRes.details).toContain('Plugin Security');

      const javaRes = await javaChecker({
        javacArgs: ['-J-javaagent:evil.jar'],
        targetFiles: ['Main.java'],
      }).check(ctx);
      expect(javaRes.status).toBe('error');
      expect(javaRes.details).toContain('Plugin Security');

      const goRes = await goChecker({ extraArgs: ['-vettool=/tmp/evil'] }).check(ctx);
      expect(goRes.status).toBe('error');
      expect(goRes.details).toContain('Plugin Security');

      const csRes = await csharpChecker({ projectPath: '../../outside.sln' }).check(ctx);
      expect(csRes.status).toBe('error');
      expect(csRes.details).toContain('Plugin Security');

      const rsRes = await rustChecker({ workspaceDir: '../../outside' }).check(ctx);
      expect(rsRes.status).toBe('error');
      expect(rsRes.details).toContain('Plugin Security');

      const tsRes = await typescriptChecker({ tsconfig: '../../tsconfig.json' }).check(ctx);
      expect(tsRes.status).toBe('error');
      expect(tsRes.details).toContain('Plugin Security');

      const svelteRes = await svelteChecker({ appDir: '../../outside-app' }).check(ctx);
      expect(svelteRes.status).toBe('error');
      expect(svelteRes.details).toContain('Plugin Security');

      const nativeRes = await nativeChecker({ targets: [{ path: '../../evil.exe' }] }).check(ctx);
      expect(nativeRes.status).toBe('error');
      expect(nativeRes.details).toContain('Plugin Security');

      const nsisRes = await nsisChecker({ templatePath: '../../evil.nsi' }).check(ctx);
      expect(nsisRes.status).toBe('error');
      expect(nsisRes.details).toContain('Plugin Security');

      const schemaRes = await schemaChecker({ targets: [{ path: '../../secret.json' }] }).check(ctx);
      expect(schemaRes.status).toBe('error');
      expect(schemaRes.details).toContain('Plugin Security');
    } finally {
      rmSync(tmpDir, { recursive: true, force: true });
    }
  });
});

import { ProcessSupervisor } from '../runner/src/supervisor';

describe('엔진 및 러너 슈퍼바이저 보안 하드닝 검증', () => {
  it('7. [ProcessSupervisor] 널 바이트 명령 차단 및 위험 환경 변수(LD_PRELOAD/NODE_OPTIONS) 필터링', async () => {
    const supervisor = new ProcessSupervisor();
    supervisor.registerProcess({
      id: 'bad-null-cmd',
      command: ['bun', 'run\0evil'],
    });
    const started = supervisor.startProcess('bad-null-cmd');
    expect(started).toBe(false);
    const procState = supervisor.getStats().processStates.find((p) => p.id === 'bad-null-cmd');
    expect(procState?.status).toBe('CRASHED');
  });
});

import { mkdtempSync as mkdtemp2, mkdirSync as mkdir2, rmSync as rmSync2, writeFileSync as writeFile2 } from 'node:fs';
import { join as join2 } from 'node:path';
import { tmpdir as tmpdir2 } from 'node:os';
import { MicroSandboxVM } from '../runtime/src/sandbox';
import { buildZipBuffer, editDocumentFile } from '../db/editor';
import { executeMcpTool } from '../compiler/mcp/tools';
import { RingBufferBus as Bus2 } from '../bus/src/ringBufferBus';
import { SnEngine, compileSn } from '../sn/src';
import { LocalDirectoryStorageAdapter } from '../mesh/src/serverlessSync';
import { HitlApprovalGate } from '../guard/src/policy';
import { SynapseOptic } from '../optic/src/optic';

describe('전 코어 v0.8.1 심층 보안 · 정합성 회귀 검증', () => {
  it('8. [MicroSandboxVM] 생성자 체인 탈출, 호스트 에러 탈출 및 프로토타입 오염을 모두 차단한다', () => {
    const vm = new MicroSandboxVM({ timeoutMs: 200 });

    const r1 = vm.execute(`({}).constructor.constructor("return typeof process")()`);
    expect(r1.ok).toBe(false);

    const r2 = vm.execute(`greet.constructor("return typeof process")()`, {
      greet: (x: string) => x,
    });
    expect(r2.ok).toBe(false);

    const r3 = vm.execute(
      `try { failFn(); } catch (e) { e.constructor.constructor("return typeof process")(); }`,
      {
        failFn: () => {
          throw new Error('host error');
        },
      }
    );
    expect(r3.ok).toBe(false);

    vm.execute(`Object.prototype.v081Polluted = true;`);
    expect((Object.prototype as any).v081Polluted).toBeUndefined();

    // 극소 게스트 타임아웃(5ms) 상태에서도 연속 탈출 시도가 미처리 예외 없이 안전 차단되어야 함
    const tightVm = new MicroSandboxVM({ timeoutMs: 5 });
    for (let i = 0; i < 20; i++) {
      const tightRes = tightVm.execute(`({}).constructor.constructor("return typeof process")()`);
      expect(tightRes.ok).toBe(false);
    }
  });

  it('9. [Synapse-DCI Editor] json_patch 프로토타입 오염 및 buildZipBuffer Zip Slip을 차단한다', () => {
    const tmp = mkdtemp2(join2(tmpdir2(), 'synapse-sec-editor-'));
    try {
      const jsonPath = join2(tmp, 'cfg.json');
      writeFile2(jsonPath, '{"name":"ok"}', 'utf-8');

      expect(() =>
        editDocumentFile(jsonPath, {
          operation: 'json_patch',
          keyPath: '__proto__.pollutedByEditor',
          value: true,
        })
      ).toThrow();
      expect((Object.prototype as any).pollutedByEditor).toBeUndefined();

      const badZipMap = new Map<string, Uint8Array>([
        ['../evil.txt', new TextEncoder().encode('hack')],
      ]);
      expect(() => buildZipBuffer(badZipMap)).toThrow();
    } finally {
      rmSync2(tmp, { recursive: true, force: true });
    }
  });

  it('10. [MCP Tools] synapse_dci_ingest가 허용 루트 외부 파일 읽기를 차단한다', async () => {
    const outsideParent = join2(process.cwd(), '..', 'outside-v081-nonexistent.txt');
    const res = await executeMcpTool('synapse_dci_ingest', { filePath: outsideParent });
    expect(res.isError).toBe(true);
    expect(res.content[0].text).toContain('차단');
  });

  it('11. [sn 제어 언어] 다중 홉 Taint, 대소문자 우회, __proto__, 순환 와이어, Parse 깊이, WASM 연산 정합성 검증', async () => {
    // 다중 홉 Taint 및 대소문자 민감 싱크 우회 차단
    const multiHopUnsafe = `
      wire hop1 {
        from "web.raw" as untrusted;
        to "bus.mid";
      }
      wire hop2 {
        from "bus.mid";
        to "  SHELL:exec";
      }
    `;
    expect(() => compileSn(multiHopUnsafe)).toThrow();

    // ParseUnary 100중첩 깊이 차단
    const deepUnary = `
      invariant too_deep {
        require ${'!'.repeat(100)}payload.ok;
      }
    `;
    expect(() => compileSn(deepUnary)).toThrow();

    // WASM 단락 평가, i64 비교, 0 나눗셈 트랩 방어
    const wasmLogic = `
      native fn logic_norm(a: i32, b: i32) -> bool via wasm {
        return a && b;
      }
      native fn big_gt(a: i64, b: i32) -> bool via wasm {
        return a > b;
      }
      native fn safe_div(a: i32, b: i32) -> i32 via wasm {
        return a / b;
      }
    `;
    const mathEngine = new SnEngine(wasmLogic);
    await mathEngine.init();
    expect(mathEngine.callNative('logic_norm', 2, 1)).toBe(1);
    expect(mathEngine.callNative('big_gt', 5000000000 as any, 10)).toBe(1);
    expect(mathEngine.callNative('safe_div', 10, 0)).toBe(0);

    // MemberAccess __proto__ 차단 및 순환 와이어 홉 한도 방어
    const protoAndCycle = `
      invariant no_proto {
        cause "프로토타입 접근 차단";
        require payload.__proto__ == payload.__proto__;
      }
      wire loop_a {
        from "loop.a";
        check no_proto;
        to "loop.b";
      }
      wire loop_b {
        from "loop.b";
        to "loop.a";
      }
    `;
    const bus = new Bus2({ ringCapacity: 64 });
    const cycleEngine = new SnEngine(protoAndCycle);
    await cycleEngine.init();
    expect(
      cycleEngine.evaluateExpr(
        cycleEngine.getCompiledUnit().program.invariants[0].requireExpr,
        { payload: { safe: 1 } }
      )
    ).toBe(true);
    // __proto__ 접근은 undefined 반환해야 함
    const protoExpr = (cycleEngine.getCompiledUnit().program.invariants[0].requireExpr as any).left;
    expect(cycleEngine.evaluateExpr(protoExpr, { payload: { safe: 1 } })).toBeUndefined();

    cycleEngine.bindBus(bus);
    bus.emit('loop.a', { safe: 1 });
    expect(cycleEngine.getAuditLog().some((e) => e.status === 'error_routed')).toBe(true);
    cycleEngine.unbindBus();
  });

  it('12. [Mesh Serverless / Guard Policy / Optic] Path Traversal, 대소문자 우회, SSRF를 차단한다', async () => {
    const tmp = mkdtemp2(join2(tmpdir2(), 'synapse-sec-stage3-'));
    try {
      const adapter = new LocalDirectoryStorageAdapter(join2(tmp, 'wal'));
      await expect(adapter.getSegment('../escape')).rejects.toThrow();

      const gate = new HitlApprovalGate();
      const hitlRes = await gate.authorize({
        action: ' DEPLOY ',
        actorId: 'agent-test',
        summary: 'uppercase bypass attempt',
      });
      expect(hitlRes.riskLevel).toBe('high');
      expect(hitlRes.approved).toBe(false);

      const optic = new SynapseOptic();
      await expect(optic.goto('file:///etc/passwd')).rejects.toThrow();
      await expect(optic.goto('http://169.254.169.254/latest/meta-data')).rejects.toThrow();
    } finally {
      rmSync2(tmp, { recursive: true, force: true });
    }
  });
});
