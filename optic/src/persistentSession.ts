/**
 * 시냅스 옵틱 영속 REPL 파이프 세션 (Persistent Session Driver)
 * - 매 액션마다 프로세스를 새로 생성(spawnSync)하고 CDP WebSocket을 재연결하는 비용을 제거
 * - 단일 prometheus-cli repl 상시 연결 파이프 위에서 명령을 직렬화 실행
 */
import { spawn } from "bun";
import { OPTIC_FLASH_DEFAULTS } from "./config";

const ANSI_ESCAPE_REGEX = /\x1b\[[0-9;]*[a-zA-Z]/g;
const PROMPT_MARKER = "prom>";

export interface PersistentSessionOptions {
  readonly binaryPath: string;
  readonly endpoint: string;
  readonly targetId?: string;
  readonly connectTimeoutMs?: number;
  readonly commandTimeoutMs?: number;
  readonly idleTtlMs?: number;
}

export interface ReplCommandResult {
  readonly ok: boolean;
  readonly output: string;
  readonly error?: string;
}

interface PendingWaiter {
  readonly resolve: (chunk: string) => void;
  readonly reject: (err: Error) => void;
  readonly timer: ReturnType<typeof setTimeout>;
}

export function stripAnsiCodes(text: string): string {
  return text.replace(ANSI_ESCAPE_REGEX, "");
}

function buildEndpointWithTarget(endpoint: string, targetId?: string): string {
  if (!targetId) return endpoint;
  if (endpoint.includes("#")) return endpoint;
  return `${endpoint}#target=${targetId}`;
}

export class OpticPersistentSession {
  private proc: ReturnType<typeof spawn> | null = null;
  private stdoutBuffer = "";
  private stderrBuffer = "";
  private currentWaiter: PendingWaiter | null = null;
  private commandQueue: Promise<unknown> = Promise.resolve();
  private idleTimer: ReturnType<typeof setTimeout> | null = null;
  private connected = false;
  private closed = false;

  private readonly connectTimeoutMs: number;
  private readonly commandTimeoutMs: number;
  private readonly idleTtlMs: number;

  constructor(private readonly options: PersistentSessionOptions) {
    this.connectTimeoutMs = options.connectTimeoutMs ?? OPTIC_FLASH_DEFAULTS.REPL_CONNECT_TIMEOUT_MS;
    this.commandTimeoutMs = options.commandTimeoutMs ?? OPTIC_FLASH_DEFAULTS.REPL_COMMAND_TIMEOUT_MS;
    this.idleTtlMs = options.idleTtlMs ?? OPTIC_FLASH_DEFAULTS.SESSION_IDLE_TTL_MS;
  }

  public isConnected(): boolean {
    return this.connected && !this.closed && this.proc !== null;
  }

  /**
   * 상시 연결 REPL 프로세스 기동 및 첫 프롬프트(배너) 대기
   */
  public async connect(): Promise<void> {
    if (this.isConnected()) return;

    this.closed = false;
    this.stdoutBuffer = "";
    this.stderrBuffer = "";

    const fullEndpoint = buildEndpointWithTarget(this.options.endpoint, this.options.targetId);
    const child = spawn({
      cmd: [this.options.binaryPath, "repl", fullEndpoint],
      stdin: "pipe",
      stdout: "pipe",
      stderr: "pipe",
    });
    this.proc = child;

    this.startStreamReaders(child);

    try {
      await this.waitForNextPrompt(this.connectTimeoutMs);
      this.connected = true;
      this.refreshIdleTimer();
    } catch (err) {
      this.forceTerminate();
      const detail = this.stderrBuffer.trim() || (err instanceof Error ? err.message : String(err));
      throw new Error(`[OpticPersistentSession] 영속 REPL 세션 연결 실패 (${fullEndpoint}): ${detail}`);
    }
  }

  /**
   * REPL 파이프 위에서 단일 라인 명령 직렬 실행
   */
  public async executeCommand(commandLine: string, timeoutMs?: number): Promise<ReplCommandResult> {
    if (/[\r\n\0]/.test(commandLine)) {
      return {
        ok: false,
        output: "",
        error: "영속 REPL 명령에는 개행 또는 널 문자를 포함할 수 없습니다.",
      };
    }

    const run = async (): Promise<ReplCommandResult> => {
      if (!this.isConnected()) {
        await this.connect();
      }
      const proc = this.proc;
      if (!proc || !proc.stdin || typeof proc.stdin === "number") {
        throw new Error("[OpticPersistentSession] REPL 표준 입력 파이프가 활성화되지 않았습니다.");
      }

      this.refreshIdleTimer();
      const waitPromise = this.waitForNextPrompt(timeoutMs ?? this.commandTimeoutMs);

      proc.stdin.write(`${commandLine}\n`);
      await proc.stdin.flush();

      const rawChunk = await waitPromise;
      this.refreshIdleTimer();
      return OpticPersistentSession.parseReplResponse(rawChunk);
    };

    const queued = this.commandQueue.then(run, run);
    this.commandQueue = queued.catch(() => undefined);

    try {
      return await queued;
    } catch (err) {
      return {
        ok: false,
        output: "",
        error: err instanceof Error ? err.message : String(err),
      };
    }
  }

  /**
   * REPL 출력 텍스트에서 성공/실패 판정
   */
  public static parseReplResponse(rawOutput: string): ReplCommandResult {
    const cleaned = stripAnsiCodes(rawOutput).trim();
    const hasFailureMarker =
      cleaned.includes("[이동 실패:") ||
      cleaned.includes("[실패:") ||
      cleaned.includes("[실패]") ||
      cleaned.includes("[캡처 실패]") ||
      cleaned.startsWith("사용법:") ||
      cleaned.startsWith("알 수 없는 명령:");

    if (hasFailureMarker) {
      return {
        ok: false,
        output: cleaned,
        error: cleaned,
      };
    }

    return {
      ok: true,
      output: cleaned,
    };
  }

  /**
   * 세션 정상 종료 및 리소스 해제
   */
  public close(): void {
    this.closed = true;
    this.connected = false;
    if (this.idleTimer) {
      clearTimeout(this.idleTimer);
      this.idleTimer = null;
    }
    if (this.currentWaiter) {
      clearTimeout(this.currentWaiter.timer);
      this.currentWaiter.reject(new Error("[OpticPersistentSession] 세션이 종료되었습니다."));
      this.currentWaiter = null;
    }
    if (this.proc) {
      try {
        if (this.proc.stdin && typeof this.proc.stdin !== "number") {
          this.proc.stdin.write("exit\n");
          this.proc.stdin.end();
        }
      } catch {}
      this.forceTerminate();
    }
  }

  private forceTerminate(): void {
    if (this.proc) {
      try {
        this.proc.kill();
      } catch {}
      this.proc = null;
    }
    this.connected = false;
  }

  private refreshIdleTimer(): void {
    if (this.idleTimer) {
      clearTimeout(this.idleTimer);
    }
    if (this.idleTtlMs > 0 && !this.closed) {
      this.idleTimer = setTimeout(() => {
        this.close();
      }, this.idleTtlMs);
      // 이벤트 루프 종료를 막지 않도록 unref 적용
      if (typeof (this.idleTimer as { unref?: () => void }).unref === "function") {
        (this.idleTimer as { unref: () => void }).unref();
      }
    }
  }

  private waitForNextPrompt(timeoutMs: number): Promise<string> {
    const immediate = this.tryConsumePromptChunk();
    if (immediate !== null) {
      return Promise.resolve(immediate);
    }

    return new Promise<string>((resolve, reject) => {
      const timer = setTimeout(() => {
        if (this.currentWaiter) {
          this.currentWaiter = null;
        }
        this.forceTerminate();
        reject(new Error(`[OpticPersistentSession] REPL 응답 대기 시간 초과 (${timeoutMs}ms)`));
      }, timeoutMs);

      this.currentWaiter = { resolve, reject, timer };
    });
  }

  private tryConsumePromptChunk(): string | null {
    const cleaned = stripAnsiCodes(this.stdoutBuffer);
    const idx = cleaned.indexOf(PROMPT_MARKER);
    if (idx === -1) return null;

    const chunk = cleaned.slice(0, idx);
    this.stdoutBuffer = cleaned.slice(idx + PROMPT_MARKER.length).replace(/^\s+/, "");
    return chunk;
  }

  private startStreamReaders(child: ReturnType<typeof spawn>): void {
    const outStream = child.stdout as ReadableStream<Uint8Array> | undefined;
    const errStream = child.stderr as ReadableStream<Uint8Array> | undefined;

    if (errStream) {
      const errReader = errStream.getReader();
      const errDecoder = new TextDecoder();
      (async () => {
        try {
          while (true) {
            const { done, value } = await errReader.read();
            if (done) break;
            this.stderrBuffer += errDecoder.decode(value, { stream: true });
            if (this.stderrBuffer.length > 8192) {
              this.stderrBuffer = this.stderrBuffer.slice(-4096);
            }
          }
        } catch {}
      })();
    }

    if (outStream) {
      const outReader = outStream.getReader();
      const outDecoder = new TextDecoder();
      (async () => {
        try {
          while (true) {
            const { done, value } = await outReader.read();
            if (done) break;
            this.stdoutBuffer += outDecoder.decode(value, { stream: true });

            if (this.currentWaiter) {
              const chunk = this.tryConsumePromptChunk();
              if (chunk !== null) {
                const w = this.currentWaiter;
                this.currentWaiter = null;
                clearTimeout(w.timer);
                w.resolve(chunk);
              }
            }
          }
        } catch (err) {
          if (this.currentWaiter) {
            const w = this.currentWaiter;
            this.currentWaiter = null;
            clearTimeout(w.timer);
            w.reject(err instanceof Error ? err : new Error(String(err)));
          }
        } finally {
          this.connected = false;
          if (this.currentWaiter) {
            const w = this.currentWaiter;
            this.currentWaiter = null;
            clearTimeout(w.timer);
            const reason = this.stderrBuffer.trim() || "프로세스가 예기치 않게 종료되었습니다.";
            w.reject(new Error(`[OpticPersistentSession] ${reason}`));
          }
        }
      })();
    }
  }
}
