/**
 * 시냅스 옵틱 Direct CDP 상시 세션 (Flashwright 스타일 Zero-Spawn WebSocket 엔진)
 * - 서브프로세스 생성(spawn) 0회, 텍스트 파싱 오버헤드 0, 고정 250ms NetworkIdle 대기 제거
 * - WeakRef 세대 격리 참조로 DOM 복제 하이재킹 방어 + CDP Input.dispatchMouseEvent로 isTrusted=true 보장
 */
import { writeFileSync } from "node:fs";
import { randomBytes } from "node:crypto";
import { OPTIC_FLASH_DEFAULTS } from "./config";

export interface DirectCdpOptions {
  readonly endpoint: string;
  readonly targetId?: string;
  readonly timeoutMs?: number;
  /** 클릭 후 적응형 정착 최소 마이크로 대기 (ms, 기본 0 = 즉시 반환, 필요 시 설정) */
  readonly postActionSettleMs?: number;
}

export interface DirectNavResult {
  readonly ok: boolean;
  readonly generation: string;
  readonly rawRefs: [string, string, string][];
  readonly error?: string;
}

export interface DirectActionResult {
  readonly ok: boolean;
  readonly tag?: string;
  readonly text?: string;
  readonly error?: string;
}

interface PendingRpc {
  readonly resolve: (result: any) => void;
  readonly reject: (err: Error) => void;
  readonly timer: ReturnType<typeof setTimeout>;
}

const SNAPSHOT_EXTRACTOR_SCRIPT = `(() => {
  const gen = Math.random().toString(16).slice(2, 10) + Date.now().toString(16);
  const storeKey = Symbol.for("__synapse_optic_weak_refs__");
  const refs = [];
  const results = [];
  const selector = [
    "button",
    "a[href]",
    "input:not([type='hidden'])",
    "textarea",
    "select",
    "[contenteditable='true']",
    "[role='button']",
    "[role='link']",
    "[role='textbox']",
    "[role='searchbox']",
    "[role='checkbox']",
    "[role='combobox']",
    "[role='menuitem']",
    "[role='tab']",
    "[role='switch']"
  ].join(",");

  const seen = new Set();
  const elements = document.querySelectorAll(selector);
  for (let i = 0; i < elements.length; i++) {
    const el = elements[i];
    if (seen.has(el)) continue;
    seen.add(el);

    const style = window.getComputedStyle(el);
    if (style.display === "none" || style.visibility === "hidden") continue;
    const rect = el.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) continue;

    const tag = el.tagName.toLowerCase();
    let role = (el.getAttribute("role") || "").toLowerCase();
    if (!role) {
      if (tag === "button" || (tag === "input" && (el.type === "button" || el.type === "submit"))) role = "button";
      else if (tag === "a") role = "link";
      else if (tag === "select") role = "combobox";
      else if (tag === "input" && el.type === "checkbox") role = "checkbox";
      else role = "textbox";
    }

    let label =
      el.getAttribute("aria-label") ||
      (el.id && document.querySelector("label[for='" + CSS.escape(el.id) + "']")?.textContent) ||
      el.getAttribute("placeholder") ||
      el.innerText ||
      el.value ||
      el.getAttribute("title") ||
      "";
    label = String(label).replace(/\\s+/g, " ").trim().slice(0, 160);

    refs.push(new WeakRef(el));
    const refId = gen + ":" + refs.length;
    results.push([refId, role, label]);
  }

  window[storeKey] = { generation: gen, refs };
  return { generation: gen, items: results };
})()`;

export class OpticDirectCdpSession {
  private ws: WebSocket | null = null;
  private nextId = 1;
  private readonly pending = new Map<number, PendingRpc>();
  private commandQueue: Promise<unknown> = Promise.resolve();
  private readonly timeoutMs: number;
  private readonly postActionSettleMs: number;
  private connected = false;

  constructor(private readonly options: DirectCdpOptions) {
    this.timeoutMs = options.timeoutMs ?? OPTIC_FLASH_DEFAULTS.REPL_COMMAND_TIMEOUT_MS;
    this.postActionSettleMs = options.postActionSettleMs ?? 0;
  }

  public isConnected(): boolean {
    return this.connected && this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }

  public async connect(): Promise<void> {
    if (this.isConnected()) return;

    const base = this.options.endpoint.replace(/#.*$/, "").replace(/\/+$/, "");
    const res = await fetch(`${base}/json/list`, {
      signal: AbortSignal.timeout(OPTIC_FLASH_DEFAULTS.REPL_CONNECT_TIMEOUT_MS),
    });
    if (!res.ok) {
      throw new Error(`[OpticDirectCdpSession] CDP 타깃 목록 조회 실패: HTTP ${res.status}`);
    }

    const targets = (await res.json()) as Array<{
      id: string;
      type: string;
      webSocketDebuggerUrl?: string;
    }>;

    const target = this.options.targetId
      ? targets.find((t) => t.id === this.options.targetId)
      : targets.find((t) => t.type === "page");

    if (!target || !target.webSocketDebuggerUrl) {
      throw new Error("[OpticDirectCdpSession] 연결 가능한 페이지 WebSocket 엔드포인트를 찾지 못했습니다.");
    }

    const ws = new WebSocket(target.webSocketDebuggerUrl);
    this.ws = ws;

    ws.addEventListener("message", (ev) => {
      try {
        const msg = JSON.parse(typeof ev.data === "string" ? ev.data : new TextDecoder().decode(ev.data));
        if (typeof msg.id === "number" && this.pending.has(msg.id)) {
          const waiter = this.pending.get(msg.id)!;
          this.pending.delete(msg.id);
          clearTimeout(waiter.timer);
          if (msg.error) {
            waiter.reject(new Error(msg.error.message || JSON.stringify(msg.error)));
          } else {
            waiter.resolve(msg.result);
          }
        }
      } catch {}
    });

    ws.addEventListener("close", () => {
      this.connected = false;
      for (const [, waiter] of this.pending) {
        clearTimeout(waiter.timer);
        waiter.reject(new Error("[OpticDirectCdpSession] CDP WebSocket 연결이 종료되었습니다."));
      }
      this.pending.clear();
    });

    await new Promise<void>((resolve, reject) => {
      const timer = setTimeout(() => {
        reject(new Error("[OpticDirectCdpSession] WebSocket 핸드셰이크 시간 초과"));
      }, OPTIC_FLASH_DEFAULTS.REPL_CONNECT_TIMEOUT_MS);

      ws.addEventListener(
        "open",
        () => {
          clearTimeout(timer);
          resolve();
        },
        { once: true }
      );
      ws.addEventListener(
        "error",
        () => {
          clearTimeout(timer);
          reject(new Error("[OpticDirectCdpSession] WebSocket 연결 오류"));
        },
        { once: true }
      );
    });

    this.connected = true;
    await Promise.all([this.send("Page.enable"), this.send("Runtime.enable")]);
  }

  public async navigateAndSnapshot(url: string): Promise<DirectNavResult> {
    return this.enqueue(async () => {
      await this.ensureConnected();
      const navRes = await this.send("Page.navigate", { url });
      if (navRes?.errorText) {
        return {
          ok: false,
          generation: "unknown",
          rawRefs: [],
          error: `네비게이션 오류: ${navRes.errorText}`,
        };
      }

      // 고정 800ms/250ms 슬립 없이 DOM readyState 즉시 반응형 폴링
      const deadline = performance.now() + this.timeoutMs;
      while (performance.now() < deadline) {
        const stateEval = await this.send("Runtime.evaluate", {
          expression: "document.readyState",
          returnByValue: true,
        });
        const readyState = stateEval?.result?.value;
        if (readyState === "interactive" || readyState === "complete") {
          break;
        }
        await new Promise((r) => setTimeout(r, 8));
      }

      const snapEval = await this.send("Runtime.evaluate", {
        expression: SNAPSHOT_EXTRACTOR_SCRIPT,
        returnByValue: true,
      });

      const val = snapEval?.result?.value as { generation: string; items: [string, string, string][] } | undefined;
      if (!val || !Array.isArray(val.items)) {
        return {
          ok: false,
          generation: randomBytes(6).toString("hex"),
          rawRefs: [],
          error: "스냅샷 노드 추출 실패",
        };
      }

      return {
        ok: true,
        generation: val.generation,
        rawRefs: val.items,
      };
    });
  }

  public async click(target: string): Promise<DirectActionResult> {
    return this.enqueue(async () => {
      await this.ensureConnected();
      const locateRes = await this.locateAndPrepareTarget(target, false);
      if (!locateRes.ok) {
        return { ok: false, error: locateRes.error };
      }

      const x = locateRes.x ?? 0;
      const y = locateRes.y ?? 0;
      const { tag, text } = locateRes;
      await this.send("Input.dispatchMouseEvent", {
        type: "mouseMoved",
        x,
        y,
        button: "none",
      });
      await this.send("Input.dispatchMouseEvent", {
        type: "mousePressed",
        x,
        y,
        button: "left",
        buttons: 1,
        clickCount: 1,
      });
      await this.send("Input.dispatchMouseEvent", {
        type: "mouseReleased",
        x,
        y,
        button: "left",
        buttons: 0,
        clickCount: 1,
      });

      if (this.postActionSettleMs > 0) {
        await new Promise((r) => setTimeout(r, this.postActionSettleMs));
      }

      return { ok: true, tag, text };
    });
  }

  public async type(target: string, text: string): Promise<DirectActionResult> {
    return this.enqueue(async () => {
      await this.ensureConnected();
      const prep = await this.focusAndClearEditable(target);
      if (!prep.ok) {
        return { ok: false, error: prep.error };
      }

      await this.send("Input.insertText", { text });
      // React/Svelte 바인딩 동기화를 위한 input/change 이벤트 통지
      await this.send("Runtime.evaluate", {
        expression: `(() => {
          const el = document.activeElement;
          if (el) {
            el.dispatchEvent(new Event("input", { bubbles: true }));
            el.dispatchEvent(new Event("change", { bubbles: true }));
          }
        })()`,
        returnByValue: true,
      });

      return { ok: true, tag: prep.tag, text };
    });
  }

  public async scroll(direction: "down" | "up", amount: number): Promise<DirectActionResult> {
    return this.enqueue(async () => {
      await this.ensureConnected();
      const deltaY = direction === "down" ? Math.abs(amount) : -Math.abs(amount);
      await this.send("Runtime.evaluate", {
        expression: `window.scrollBy(0, ${Number(deltaY) || 0})`,
        returnByValue: true,
      });
      return { ok: true };
    });
  }

  public async screenshot(outputPath: string): Promise<DirectActionResult> {
    return this.enqueue(async () => {
      await this.ensureConnected();
      const shot = await this.send("Page.captureScreenshot", { format: "png" });
      if (!shot?.data) {
        return { ok: false, error: "스크린샷 데이터 캡처 실패" };
      }
      writeFileSync(outputPath, Buffer.from(shot.data, "base64"));
      return { ok: true };
    });
  }

  public close(): void {
    this.connected = false;
    if (this.ws) {
      try {
        this.ws.close();
      } catch {}
      this.ws = null;
    }
    for (const [, waiter] of this.pending) {
      clearTimeout(waiter.timer);
      waiter.reject(new Error("[OpticDirectCdpSession] 세션이 닫혔습니다."));
    }
    this.pending.clear();
  }

  private async ensureConnected(): Promise<void> {
    if (!this.isConnected()) {
      await this.connect();
    }
  }

  private enqueue<T>(fn: () => Promise<T>): Promise<T> {
    const next = this.commandQueue.then(fn, fn);
    this.commandQueue = next.catch(() => undefined);
    return next;
  }

  private send(method: string, params: Record<string, unknown> = {}): Promise<any> {
    const ws = this.ws;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      return Promise.reject(new Error("[OpticDirectCdpSession] 열린 WebSocket 연결이 없습니다."));
    }
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`[OpticDirectCdpSession] CDP 요청 시간 초과 (${method}, ${this.timeoutMs}ms)`));
      }, this.timeoutMs);

      this.pending.set(id, { resolve, reject, timer });
      ws.send(JSON.stringify({ id, method, params }));
    });
  }

  private async locateAndPrepareTarget(
    target: string,
    requireEditable: boolean
  ): Promise<{ ok: boolean; x?: number; y?: number; tag?: string; text?: string; error?: string }> {
    const evalRes = await this.send("Runtime.evaluate", {
      expression: `((rawTarget, needEdit) => {
        const store = window[Symbol.for("__synapse_optic_weak_refs__")];
        let el = null;

        const refMatch = rawTarget.match(/^@?([a-zA-Z0-9]+):(\\d+)$/);
        if (refMatch) {
          if (!store || store.generation !== refMatch[1]) {
            return { ok: false, error: "STALE_GENERATION: 만료된 스냅샷 참조입니다." };
          }
          const idx = Number(refMatch[2]) - 1;
          const wr = store.refs[idx];
          el = wr ? wr.deref() : null;
          if (!el || !el.isConnected) {
            return { ok: false, error: "DETACHED_NODE: 요소가 DOM에서 분리되었습니다." };
          }
        } else {
          try {
            const matches = document.querySelectorAll(rawTarget);
            if (matches.length > 1) {
              return { ok: false, error: "AMBIGUOUS_SELECTOR: 복수의 요소가 일치합니다." };
            }
            el = matches[0] || null;
          } catch {
            el = null;
          }
        }

        if (!el) {
          return { ok: false, error: "ELEMENT_NOT_FOUND: 요소를 찾을 수 없습니다." };
        }
        if (el.disabled) {
          return { ok: false, error: "ELEMENT_DISABLED: 비활성화된 요소입니다." };
        }
        if (needEdit && el.readOnly) {
          return { ok: false, error: "ELEMENT_READONLY: 읽기 전용 요소입니다." };
        }

        let rect = el.getBoundingClientRect();
        if (
          rect.bottom <= 0 ||
          rect.top >= window.innerHeight ||
          rect.right <= 0 ||
          rect.left >= window.innerWidth
        ) {
          el.scrollIntoView({ block: "center", inline: "center", behavior: "instant" });
          rect = el.getBoundingClientRect();
        }
        if (rect.width <= 0 || rect.height <= 0) {
          return { ok: false, error: "ELEMENT_HIDDEN: 화면에 표시되지 않는 요소입니다." };
        }

        const x = rect.left + rect.width / 2;
        const y = rect.top + rect.height / 2;
        const hit = document.elementFromPoint(x, y);
        if (hit && hit !== el && !el.contains(hit) && !hit.contains(el)) {
          return { ok: false, error: "ELEMENT_COVERED: 다른 요소에 가려져 있습니다." };
        }

        return {
          ok: true,
          x,
          y,
          tag: el.tagName.toLowerCase(),
          text: String(el.innerText || el.value || "").slice(0, 80)
        };
      })(${JSON.stringify(target)}, ${requireEditable})`,
      returnByValue: true,
    });

    const val = evalRes?.result?.value;
    if (!val || !val.ok) {
      return { ok: false, error: val?.error || "ELEMENT_NOT_FOUND" };
    }
    return val;
  }

  private async focusAndClearEditable(
    target: string
  ): Promise<{ ok: boolean; tag?: string; error?: string }> {
    const evalRes = await this.send("Runtime.evaluate", {
      expression: `((rawTarget) => {
        const store = window[Symbol.for("__synapse_optic_weak_refs__")];
        let el = null;

        const refMatch = rawTarget.match(/^@?([a-zA-Z0-9]+):(\\d+)$/);
        if (refMatch) {
          if (!store || store.generation !== refMatch[1]) {
            return { ok: false, error: "STALE_GENERATION: 만료된 스냅샷 참조입니다." };
          }
          const idx = Number(refMatch[2]) - 1;
          const wr = store.refs[idx];
          el = wr ? wr.deref() : null;
        } else {
          try {
            const matches = document.querySelectorAll(rawTarget);
            if (matches.length === 1) el = matches[0];
          } catch {}
        }

        if (!el || !el.isConnected) {
          return { ok: false, error: "ELEMENT_NOT_FOUND: 입력 요소를 찾을 수 없습니다." };
        }
        if (el.disabled || el.readOnly) {
          return { ok: false, error: "ELEMENT_NOT_EDITABLE: 편집할 수 없는 요소입니다." };
        }

        const tag = el.tagName.toLowerCase();
        const isEditable =
          tag === "input" ||
          tag === "textarea" ||
          el.isContentEditable ||
          el.getAttribute("contenteditable") === "true";
        if (!isEditable) {
          return { ok: false, error: "ELEMENT_NOT_EDITABLE: 입력 가능한 요소가 아닙니다." };
        }

        el.focus();
        if ("value" in el) {
          el.value = "";
        } else {
          el.textContent = "";
        }
        return { ok: true, tag };
      })(${JSON.stringify(target)})`,
      returnByValue: true,
    });

    const val = evalRes?.result?.value;
    if (!val || !val.ok) {
      return { ok: false, error: val?.error || "ELEMENT_NOT_FOUND" };
    }
    return val;
  }
}
