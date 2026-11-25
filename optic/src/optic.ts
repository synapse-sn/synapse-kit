/**
 * 시냅스 옵틱 (Synapse Optic) 통합 브라우저 파사드
 * - Direct CDP 상시 WebSocket 엔진, Rust 프로메테우스 REPL 파이프, 유사 로케이터 진단 및 버스/가드 연동
 */
import { OPTIC_DEFAULTS, type OpticConfig } from "./config";
import type { OpticSnapshot, OpticActionResult, ReferenceNode, OpticStatus } from "./types";
import { OpticDriver } from "./driver";
import { A11yParser } from "./a11yParser";
import { LocatorDiagnostics } from "./locatorDiagnostics";
import { OpticPersistentSession, stripAnsiCodes } from "./persistentSession";
import { OpticDirectCdpSession } from "./directCdpSession";

export interface SynapseOpticOptions extends OpticConfig {
  /** 시냅스 이벤트 버스 연동 (선택적) */
  readonly bus?: { emit: (event: string, payload: any) => void };
  /** 시냅스 보안 가드 연동 (선택적) */
  readonly guard?: { scanSecret: (text: string) => { hasSecret: boolean; match?: string } };
}

const MAX_SNAPSHOT_CACHE_SIZE = 256;

function validateOpticUrl(rawUrl: string): string {
  const trimmed = rawUrl.trim();
  if (!trimmed || trimmed.startsWith("-") || trimmed.includes("\0")) {
    throw new Error(`[SynapseOptic Security] 안전하지 않은 URL 또는 플래그 인젝션입니다: ${rawUrl}`);
  }
  let parsed: URL;
  try {
    parsed = new URL(trimmed);
  } catch {
    throw new Error(`[SynapseOptic Security] 유효하지 않은 URL 형식입니다: ${rawUrl}`);
  }
  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
    throw new Error(`[SynapseOptic Security] 허용되지 않는 프로토콜입니다(${parsed.protocol}). http/https만 허용됩니다.`);
  }
  if (parsed.username || parsed.password) {
    throw new Error("[SynapseOptic Security] URL에 인증 정보(userinfo)를 포함할 수 없습니다.");
  }
  const host = parsed.hostname.toLowerCase().replace(/^\[|\]$/g, "");
  if (
    !host ||
    host === "0.0.0.0" ||
    host === "::" ||
    host.startsWith("169.254.") ||
    host.startsWith("fe80:") ||
    host === "metadata.google.internal" ||
    host === "metadata.azure.com" ||
    host === "100.100.100.200"
  ) {
    throw new Error(`[SynapseOptic Security] 차단된 내부/메타데이터 호스트입니다: ${host || "(empty)"}`);
  }
  return parsed.toString();
}

export class SynapseOptic {
  public readonly driver: OpticDriver;
  private endpoint: string;
  private currentUrl?: string;
  private ownedBrowserProc?: ReturnType<typeof import("bun").spawn>;
  private directSession: OpticDirectCdpSession | null = null;
  private replSession: OpticPersistentSession | null = null;
  private usePersistentSession: boolean;
  private readonly sessionBackend: "direct_cdp" | "repl_pipe";
  private readonly enableDiagnostics: boolean;
  private lastSnapshot: OpticSnapshot | null = null;
  private readonly snapshotCache = new Map<string, { snapshot: OpticSnapshot; timestamp: number }>();

  constructor(private readonly options: SynapseOpticOptions = {}) {
    this.endpoint = options.endpoint ?? OPTIC_DEFAULTS.DEFAULT_ENDPOINT;
    this.driver = new OpticDriver(options);
    this.usePersistentSession = options.persistentSession ?? false;
    this.sessionBackend = options.sessionBackend ?? "direct_cdp";
    this.enableDiagnostics = options.enableDiagnostics ?? true;
  }

  public getStatus(): OpticStatus {
    const active =
      this.sessionBackend === "direct_cdp"
        ? (this.directSession?.isConnected() ?? false)
        : (this.replSession?.isConnected() ?? false);
    return {
      connected: this.driver.isBinaryAvailable() || active,
      endpoint: this.endpoint,
      targetId: this.options.targetId,
      currentUrl: this.currentUrl,
      actionTimeoutMs: this.options.timeoutMs ?? OPTIC_DEFAULTS.DEFAULT_ACTION_TIMEOUT_MS,
      persistentSessionActive: active,
    };
  }

  public setActiveSnapshot(snapshot: OpticSnapshot | null): void {
    this.lastSnapshot = snapshot;
  }

  public getActiveSnapshot(): OpticSnapshot | null {
    return this.lastSnapshot;
  }

  public async launch(port = 0): Promise<string> {
    const { proc, endpointPromise } = this.driver.launchBrowser(port);
    this.ownedBrowserProc = proc;
    const endpoint = await endpointPromise;
    this.endpoint = endpoint;
    if (this.usePersistentSession) {
      await this.connectPersistentSession();
    }
    this.emitBus("optic.browser.launched", { endpoint });
    return endpoint;
  }

  /**
   * 영속 세션 연결 ("direct_cdp" 상시 WebSocket 또는 "repl_pipe" 네이티브 파이프)
   */
  public async connectPersistentSession(): Promise<void> {
    if (this.sessionBackend === "direct_cdp") {
      if (this.directSession && this.directSession.isConnected()) {
        this.usePersistentSession = true;
        return;
      }
      if (this.directSession) {
        this.directSession.close();
      }
      this.directSession = new OpticDirectCdpSession({
        endpoint: this.endpoint,
        targetId: this.options.targetId,
        timeoutMs: this.options.timeoutMs,
      });
      await this.directSession.connect();
      this.usePersistentSession = true;
      this.emitBus("optic.session.connected", { endpoint: this.endpoint, backend: "direct_cdp" });
      return;
    }

    const binPath = this.driver.resolveBinaryPath();
    if (!binPath) {
      throw new Error("[SynapseOptic] 프로메테우스 네이티브 바이너리를 찾을 수 없어 REPL 세션을 시작할 수 없습니다.");
    }
    if (this.replSession && this.replSession.isConnected()) {
      this.usePersistentSession = true;
      return;
    }
    if (this.replSession) {
      this.replSession.close();
    }
    this.replSession = new OpticPersistentSession({
      binaryPath: binPath,
      endpoint: this.endpoint,
      targetId: this.options.targetId,
      commandTimeoutMs: this.options.timeoutMs,
    });
    await this.replSession.connect();
    this.usePersistentSession = true;
    this.emitBus("optic.session.connected", { endpoint: this.endpoint, backend: "repl_pipe" });
  }

  public disconnectPersistentSession(): void {
    if (this.directSession) {
      this.directSession.close();
      this.directSession = null;
    }
    if (this.replSession) {
      this.replSession.close();
      this.replSession = null;
    }
    this.usePersistentSession = false;
  }

  public async goto(url: string, options: { forceRefresh?: boolean; ttlMs?: number } = {}): Promise<OpticSnapshot> {
    const safeUrl = validateOpticUrl(url);
    const ttlMs = options.ttlMs ?? 120_000;
    const now = performance.now();
    const cached = this.snapshotCache.get(safeUrl);

    if (!options.forceRefresh && cached && now - cached.timestamp < ttlMs) {
      this.lastSnapshot = cached.snapshot;
      this.emitBus("optic.navigated.cache_hit", { url: safeUrl, durationMs: 0 });
      return {
        ...cached.snapshot,
        durationMs: 0.05,
      };
    }

    const start = performance.now();
    this.emitBus("optic.navigating", { url: safeUrl });

    let snapshot: OpticSnapshot;

    if (this.usePersistentSession && this.sessionBackend === "direct_cdp") {
      if (!this.directSession || !this.directSession.isConnected()) {
        await this.connectPersistentSession();
      }
      const navRes = await this.directSession!.navigateAndSnapshot(safeUrl);
      const durationMs = performance.now() - start;
      if (!navRes.ok) {
        this.emitBus("optic.nav.failed", { url: safeUrl, error: navRes.error });
        throw new Error(`[SynapseOptic] 페이지 이동 실패 (${safeUrl}): ${navRes.error}`);
      }
      const nodes = A11yParser.formatReferenceSnapshot(navRes.rawRefs);
      snapshot = {
        generation: navRes.generation,
        totalNodes: nodes.length,
        formattedPrompt: A11yParser.formatSnapshotSummary(nodes),
        nodes,
        durationMs,
      };
    } else if (this.usePersistentSession && this.sessionBackend === "repl_pipe") {
      if (!this.replSession || !this.replSession.isConnected()) {
        await this.connectPersistentSession();
      }
      const replRes = await this.replSession!.executeCommand(`nav ${safeUrl}`, this.options.timeoutMs);
      const durationMs = performance.now() - start;
      if (!replRes.ok) {
        this.emitBus("optic.nav.failed", { url: safeUrl, error: replRes.error });
        throw new Error(`[SynapseOptic] 페이지 이동 실패 (${safeUrl}): ${replRes.error}`);
      }
      snapshot = this.parseNavSnapshot(replRes.output, durationMs);
    } else {
      const res = this.driver.runCommand(["nav", safeUrl, this.endpoint]);
      const durationMs = performance.now() - start;
      if (!res.ok) {
        this.emitBus("optic.nav.failed", { url: safeUrl, error: res.stderr });
        throw new Error(`[SynapseOptic] 페이지 이동 실패 (${safeUrl}): ${res.stderr}`);
      }
      snapshot = this.parseNavSnapshot(res.stdout, durationMs);
    }

    this.currentUrl = safeUrl;
    this.lastSnapshot = snapshot;

    if (this.snapshotCache.size >= MAX_SNAPSHOT_CACHE_SIZE && !this.snapshotCache.has(safeUrl)) {
      const oldestKey = this.snapshotCache.keys().next().value;
      if (oldestKey !== undefined) {
        this.snapshotCache.delete(oldestKey);
      }
    }
    this.snapshotCache.set(safeUrl, { snapshot, timestamp: now });
    this.emitBus("optic.navigated", { url: safeUrl, totalNodes: snapshot.totalNodes, durationMs: snapshot.durationMs });
    return snapshot;
  }

  public async gotoBatch(urls: readonly string[], options: { forceRefresh?: boolean } = {}): Promise<OpticSnapshot[]> {
    return Promise.all(urls.map((url) => this.goto(url, options)));
  }

  public clearCache(): void {
    this.snapshotCache.clear();
  }

  public getCacheStats(): { size: number; cachedUrls: readonly string[] } {
    return {
      size: this.snapshotCache.size,
      cachedUrls: Array.from(this.snapshotCache.keys()),
    };
  }

  public async preload(urls: readonly string[]): Promise<void> {
    await Promise.allSettled(urls.map((url) => this.goto(url)));
  }

  public warmup(): { ready: boolean; binaryAvailable: boolean; endpoint: string } {
    const binaryAvailable = this.driver.isBinaryAvailable();
    return {
      ready: binaryAvailable,
      binaryAvailable,
      endpoint: this.endpoint,
    };
  }

  public async click(target: string): Promise<OpticActionResult> {
    const resolved = LocatorDiagnostics.resolveTargetReference(target, this.lastSnapshot);
    return this.executeAction(
      "click",
      ["click", resolved, this.endpoint],
      target,
      undefined,
      `click ${resolved}`,
      () => this.directSession!.click(resolved)
    );
  }

  public async hover(target: string): Promise<OpticActionResult> {
    const resolved = LocatorDiagnostics.resolveTargetReference(target, this.lastSnapshot);
    return this.executeAction("hover", ["hover", resolved, this.endpoint], target);
  }

  public async doubleClick(target: string): Promise<OpticActionResult> {
    const resolved = LocatorDiagnostics.resolveTargetReference(target, this.lastSnapshot);
    return this.executeAction("double_click", ["double-click", resolved, this.endpoint], target);
  }

  public async rightClick(target: string): Promise<OpticActionResult> {
    const resolved = LocatorDiagnostics.resolveTargetReference(target, this.lastSnapshot);
    return this.executeAction("right_click", ["right-click", resolved, this.endpoint], target);
  }

  public async press(key: string): Promise<OpticActionResult> {
    return this.executeAction("press", ["press", key, this.endpoint], key);
  }

  public async type(target: string, text: string): Promise<OpticActionResult> {
    this.verifySecretGuard(text);
    const resolved = LocatorDiagnostics.resolveTargetReference(target, this.lastSnapshot);
    const replSafe = !/[\r\n]/.test(text) && !/\s{2,}/.test(text) && text.trim() === text && text.length > 0;
    const replCmd = replSafe && !/\s/.test(resolved) ? `type ${resolved} ${text}` : undefined;
    return this.executeAction(
      "type",
      ["type", resolved, text, this.endpoint],
      target,
      text,
      replCmd,
      () => this.directSession!.type(resolved, text)
    );
  }

  public async typeNative(target: string, text: string): Promise<OpticActionResult> {
    this.verifySecretGuard(text);
    const resolved = LocatorDiagnostics.resolveTargetReference(target, this.lastSnapshot);
    return this.executeAction("type_native", ["type-native", resolved, text, this.endpoint], target, text);
  }

  public async scroll(direction: "down" | "up" = "down", amount = OPTIC_DEFAULTS.DEFAULT_SCROLL_AMOUNT): Promise<OpticActionResult> {
    return this.executeAction(
      "scroll",
      ["scroll", direction, String(amount), this.endpoint],
      undefined,
      undefined,
      `scroll ${direction} ${amount}`,
      () => this.directSession!.scroll(direction, amount)
    );
  }

  public async screenshot(outputPath = "screenshot.png"): Promise<OpticActionResult> {
    const replCmd = !/\s/.test(outputPath) ? `shot ${outputPath}` : undefined;
    return this.executeAction(
      "screenshot",
      ["shot", outputPath, this.endpoint],
      undefined,
      undefined,
      replCmd,
      () => this.directSession!.screenshot(outputPath)
    );
  }

  public close(): void {
    this.disconnectPersistentSession();
    if (this.ownedBrowserProc) {
      try {
        this.ownedBrowserProc.kill();
      } catch {}
      this.ownedBrowserProc = undefined;
    }
    this.emitBus("optic.closed", {});
  }

  private parseNavSnapshot(stdout: string, durationMs: number): OpticSnapshot {
    const rawRefs: [string, string, string][] = [];
    const lines = stripAnsiCodes(stdout).split("\n");
    const normalizeRole = (rawRole: string): string => {
      if (rawRole === "버튼") return "button";
      if (rawRole === "입력") return "textbox";
      if (rawRole === "링크") return "link";
      return rawRole;
    };

    for (const line of lines) {
      const replMatch = line.match(/\[@([^\]]+)\]\s+\[([^\]]+)\]\s*(.*)/);
      if (replMatch) {
        const ref = replMatch[1].trim();
        const role = normalizeRole(replMatch[2].trim());
        const label = replMatch[3].trim();
        rawRefs.push([ref, role, label]);
        continue;
      }

      const cliMatch = line.match(/\[([^\]]+)\]\s+\[@([^\]]+)\]\s+(.*)/);
      if (cliMatch) {
        const role = normalizeRole(cliMatch[1].trim());
        const ref = cliMatch[2].trim();
        const label = cliMatch[3].trim();
        rawRefs.push([ref, role, label]);
      }
    }

    const nodes: ReferenceNode[] = A11yParser.formatReferenceSnapshot(rawRefs);
    const formattedPrompt = A11yParser.formatSnapshotSummary(nodes);
    const generation = rawRefs[0] ? rawRefs[0][0].split(":")[0] : "unknown";

    return {
      generation,
      totalNodes: nodes.length,
      formattedPrompt,
      nodes,
      durationMs,
    };
  }

  private async executeAction(
    action: string,
    cliArgs: string[],
    requestedTarget?: string,
    text?: string,
    replCommand?: string,
    directRunner?: () => Promise<{ ok: boolean; error?: string }>
  ): Promise<OpticActionResult> {
    const start = performance.now();
    let ok = false;
    let errText: string | undefined;
    let usedPersistentSession = false;

    if (this.usePersistentSession && this.sessionBackend === "direct_cdp" && directRunner) {
      try {
        if (!this.directSession || !this.directSession.isConnected()) {
          await this.connectPersistentSession();
        }
        const directRes = await directRunner();
        ok = directRes.ok;
        errText = directRes.ok ? undefined : directRes.error;
        usedPersistentSession = true;
      } catch (err) {
        ok = false;
        errText = err instanceof Error ? err.message : String(err);
      }
    } else if (
      this.usePersistentSession &&
      this.sessionBackend === "repl_pipe" &&
      replCommand &&
      !/\s/.test(cliArgs[1] ?? "")
    ) {
      try {
        if (!this.replSession || !this.replSession.isConnected()) {
          await this.connectPersistentSession();
        }
        const replRes = await this.replSession!.executeCommand(replCommand, this.options.timeoutMs);
        ok = replRes.ok;
        errText = replRes.ok ? undefined : replRes.error;
        usedPersistentSession = true;
      } catch (err) {
        ok = false;
        errText = err instanceof Error ? err.message : String(err);
      }
    } else {
      const res = this.driver.runCommand(cliArgs);
      ok = res.ok;
      errText = res.ok ? undefined : res.stderr;
    }

    const durationMs = performance.now() - start;

    let suggestions: OpticActionResult["suggestions"];
    let diagnosticHint: string | undefined;
    let finalError = errText;

    if (!ok && this.enableDiagnostics && requestedTarget && action !== "press") {
      const diag = LocatorDiagnostics.diagnose(requestedTarget, this.lastSnapshot);
      suggestions = diag.suggestions;
      diagnosticHint = diag.hint;
      if (this.lastSnapshot && this.lastSnapshot.nodes.length > 0) {
        finalError = errText ? `${errText} | ${diag.formattedMessage}` : diag.formattedMessage;
      }
    }

    const result: OpticActionResult = {
      ok,
      action,
      target: requestedTarget,
      text,
      durationMs,
      error: finalError,
      suggestions,
      diagnosticHint,
      usedPersistentSession,
    };

    if (ok) {
      this.emitBus(`optic.action.${action}.success`, result);
    } else {
      this.emitBus(`optic.action.${action}.failed`, result);
    }

    return result;
  }

  private verifySecretGuard(text: string): void {
    if (this.options.guard) {
      const check = this.options.guard.scanSecret(text);
      if (check.hasSecret) {
        throw new Error(`[SynapseOptic Guard] 보안 경고: 민감한 자격증명(${check.match})이 브라우저 입력에 감지되어 전송이 차단되었습니다.`);
      }
    }
  }

  private emitBus(event: string, payload: any): void {
    if (this.options.bus) {
      try {
        this.options.bus.emit(event, payload);
      } catch {}
    }
  }
}
