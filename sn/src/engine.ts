/**
 * 시냅스 제어 플레인 런타임 엔진 (Synapse Control Plane Engine)
 * - 컴파일된 .syn 유닛의 WASM 네이티브 함수(JIT 기계어)를 인스턴스화합니다.
 * - RingBufferBus와 직접 결합해 와이어 라우팅, Taint/Sanitize, 불변식 검증, 에러 폴백을 집행합니다.
 */
export type ControlUnsubscribe = () => void;

export interface ControlBusLike {
  subscribe(
    topicPattern: string,
    handler: (msg: { payload: any }) => void | Promise<void>,
  ): ControlUnsubscribe;
  emit(topic: string, payload?: any, sender?: string): number;
}
import { compileSn } from "./codegen";
import { SN_DEFAULTS } from "./config";
import {
  SnError,
  type CompiledSnUnit,
  type ControlAuditEntry,
  type Expr,
  type InvariantDecl,
  type NativeFfiHandler,
  type SanitizerFn,
} from "./types";

export class SnEngine {
  private readonly unit: CompiledSnUnit;
  private readonly wasmExports = new Map<string, (...args: any[]) => any>();
  private readonly ffiHandlers = new Map<string, NativeFfiHandler>();
  private readonly sanitizers = new Map<string, SanitizerFn>();
  private readonly invariantsByName = new Map<string, InvariantDecl>();
  private readonly auditLog: ControlAuditEntry[] = [];
  private readonly unsubscribes: ControlUnsubscribe[] = [];
  private initialized = false;

  constructor(sourceOrUnit: string | CompiledSnUnit) {
    this.unit =
      typeof sourceOrUnit === "string"
        ? compileSn(sourceOrUnit, { strict: true })
        : sourceOrUnit;

    for (const inv of this.unit.program.invariants) {
      this.invariantsByName.set(inv.name, inv);
    }
  }

  /**
   * 모든 WASM 네이티브 모듈을 호스트 기계어로 인스턴스화합니다.
   */
  public async init(): Promise<void> {
    if (this.initialized) return;
    for (const [name, compiled] of this.unit.wasmFunctions.entries()) {
      const result = await WebAssembly.instantiate(compiled.wasmBytes, {});
      const instance =
        result instanceof WebAssembly.Instance ? result : result.instance;
      const fn = instance.exports[name];
      if (typeof fn !== "function") {
        throw new SnError(
          `[sn] WASM 인스턴스에서 내보낸 함수 '${name}'을(를) 찾을 수 없습니다.`,
          "ENGINE_WASM_EXPORT_MISSING",
        );
      }
      this.wasmExports.set(name, fn as (...args: any[]) => any);
    }
    this.initialized = true;
  }

  public registerSanitizer(name: string, fn: SanitizerFn): void {
    this.sanitizers.set(name, fn);
  }

  public registerFfiHandler(name: string, fn: NativeFfiHandler): void {
    this.ffiHandlers.set(name, fn);
  }

  public callNative(name: string, ...args: number[]): number | bigint {
    const wasmFn = this.wasmExports.get(name);
    if (wasmFn) {
      return wasmFn(...args);
    }
    const ffiFn = this.ffiHandlers.get(name);
    if (ffiFn) {
      return ffiFn(...args);
    }
    throw new SnError(
      `[sn] 등록되지 않은 네이티브 함수 호출입니다: '${name}'`,
      "ENGINE_UNKNOWN_NATIVE_FN",
    );
  }

  /**
   * eval() 없이 순수 AST 워커로 불변식 표현식을 안전하게 평가합니다.
   */
  public evaluateExpr(expr: Expr, scope: Record<string, any>): any {
    switch (expr.kind) {
      case "NumberLiteral":
      case "BoolLiteral":
      case "StringLiteral":
        return expr.value;
      case "Identifier":
        return scope[expr.name];
      case "MemberAccess": {
        const obj = this.evaluateExpr(expr.object, scope);
        if (obj === null || obj === undefined || typeof obj !== "object") {
          return undefined;
        }
        return obj[expr.property];
      }
      case "Unary": {
        const val = this.evaluateExpr(expr.operand, scope);
        return expr.op === "-" ? -Number(val) : !val;
      }
      case "Binary": {
        if (expr.op === "&&") {
          return Boolean(this.evaluateExpr(expr.left, scope)) &&
            Boolean(this.evaluateExpr(expr.right, scope));
        }
        if (expr.op === "||") {
          return Boolean(this.evaluateExpr(expr.left, scope)) ||
            Boolean(this.evaluateExpr(expr.right, scope));
        }
        const l = this.evaluateExpr(expr.left, scope);
        const r = this.evaluateExpr(expr.right, scope);
        switch (expr.op) {
          case "+":
            return l + r;
          case "-":
            return l - r;
          case "*":
            return l * r;
          case "/":
            return r === 0 ? 0 : l / r;
          case "%":
            return r === 0 ? 0 : l % r;
          case "==":
            return l === r;
          case "!=":
            return l !== r;
          case "<":
            return l < r;
          case "<=":
            return l <= r;
          case ">":
            return l > r;
          case ">=":
            return l >= r;
        }
      }
      case "Call": {
        const evaluatedArgs = expr.args.map((a) => this.evaluateExpr(a, scope));
        return this.callNative(expr.callee, ...evaluatedArgs);
      }
    }
  }

  private recordAudit(entry: ControlAuditEntry): void {
    if (this.auditLog.length >= SN_DEFAULTS.MAX_AUDIT_ENTRIES) {
      this.auditLog.shift();
    }
    this.auditLog.push(entry);
  }

  /**
   * 컴파일된 모든 와이어를 RingBufferBus(또는 호환 이벤트 버스)에 바인딩합니다.
   */
  public bindBus(bus: ControlBusLike): void {
    this.unbindBus();

    for (const wire of this.unit.program.wires) {
      const unsub = bus.subscribe(wire.fromTopic, (msg) => {
        try {
          let payload = msg.payload;

          if (wire.sanitizer) {
            const sanitizerFn = this.sanitizers.get(wire.sanitizer);
            if (!sanitizerFn) {
              throw new SnError(
                `[sn] 와이어 '${wire.name}'의 새니타이저 '${wire.sanitizer}'가 등록되지 않았습니다.`,
                "ENGINE_MISSING_SANITIZER",
              );
            }
            payload = sanitizerFn(payload);
          }

          for (const checkName of wire.checks) {
            const inv = this.invariantsByName.get(checkName);
            if (!inv) continue;
            const passed = Boolean(
              this.evaluateExpr(inv.requireExpr, { payload }),
            );
            if (!passed) {
              const reason = `[${inv.name}] ${inv.cause || "불변식 위반"}`;
              this.recordAudit({
                timestampMs: Date.now(),
                wireName: wire.name,
                fromTopic: wire.fromTopic,
                toTarget: wire.toTarget,
                status: "blocked_invariant",
                reason,
              });
              if (wire.onErrorTopic) {
                bus.emit(
                  wire.onErrorTopic,
                  { wire: wire.name, reason, payload },
                  `synapse-lang:${wire.name}`,
                );
              }
              return;
            }
          }

          bus.emit(wire.toTarget, payload, `synapse-lang:${wire.name}`);
          this.recordAudit({
            timestampMs: Date.now(),
            wireName: wire.name,
            fromTopic: wire.fromTopic,
            toTarget: wire.toTarget,
            status: "delivered",
          });
        } catch (err) {
          const reason = err instanceof Error ? err.message : String(err);
          this.recordAudit({
            timestampMs: Date.now(),
            wireName: wire.name,
            fromTopic: wire.fromTopic,
            toTarget: wire.toTarget,
            status: "error_routed",
            reason,
          });
          if (wire.onErrorTopic) {
            bus.emit(
              wire.onErrorTopic,
              { wire: wire.name, reason, payload: msg.payload },
              `synapse-lang:${wire.name}`,
            );
          }
        }
      });

      this.unsubscribes.push(unsub);
    }
  }

  public unbindBus(): void {
    for (const unsub of this.unsubscribes) {
      unsub();
    }
    this.unsubscribes.length = 0;
  }

  public getCompiledUnit(): CompiledSnUnit {
    return this.unit;
  }

  public getAuditLog(): readonly ControlAuditEntry[] {
    return this.auditLog;
  }
}
