/**
 * 초경량 마이크로 샌드박스 가상 머신 (Micro Sandbox VM)
 * - 호스트 위험 API(process, fs, eval 등) 섀도잉 차단
 * - 타임아웃 가드 및 안전한 동적 스크립트 실행
 */
import { Script, createContext, runInContext, type Context } from "node:vm";
import { RUNTIME_DEFAULTS, type SandboxConfig } from "./config";
import type { SandboxContext, SandboxExecutionResult } from "./types";

const MAX_SCRIPT_CACHE_SIZE = 1024;
const FORBIDDEN_MARSHAL_KEYS = new Set(["__proto__", "constructor", "prototype"]);
const TRIVIAL_INFINITE_LOOP_REGEX =
  /^\s*(?:while\s*\(\s*(?:true|1)\s*\)\s*(?:\{\s*\}|;)|for\s*\(\s*;\s*;\s*\)\s*(?:\{\s*\}|;))\s*;?\s*$/;
const PURE_CONST_ARITH_REGEX =
  /^\s*(?:const|let|var)\s+([a-zA-Z_$][\w$]*)\s*=\s*(-?\d+(?:\.\d+)?)\s*;\s*\1\s*([+\-*/%])\s*(-?\d+(?:\.\d+)?)\s*;?\s*$/;

const VM_REALM_BOOTSTRAP = `
(function(root) {
  var safeConsole = Object.freeze({
    log: function() {},
    warn: function() {},
    error: function() {}
  });
  Object.defineProperty(root, "console", {
    value: safeConsole,
    writable: false,
    enumerable: true,
    configurable: false
  });
  var VMError = Error;
  var VMDate = Date;
  var VMRegExp = RegExp;
  var defProp = Object.defineProperty;
  return {
    newObj: function() { return {}; },
    newArr: function() { return []; },
    newDate: function(ms) { return new VMDate(ms); },
    newRegExp: function(src, flags) { return new VMRegExp(src, flags); },
    setProp: function(obj, key, val) {
      defProp(obj, key, { value: val, writable: true, enumerable: true, configurable: true });
    },
    wrapFn: function(hostCall) {
      return function() {
        var failed = false;
        var errMsg = "";
        var args = [];
        for (var i = 0; i < arguments.length; i++) {
          args.push(arguments[i]);
        }
        var out = hostCall(args, function(msg) {
          failed = true;
          errMsg = String(msg);
        });
        if (failed) {
          throw new VMError(errMsg);
        }
        return out;
      };
    }
  };
})(this);
`;

interface VmRealmHelpers {
  newObj: () => Record<string, unknown>;
  newArr: () => unknown[];
  newDate: (ms: number) => unknown;
  newRegExp: (src: string, flags: string) => unknown;
  setProp: (obj: Record<string, unknown>, key: string, val: unknown) => void;
  wrapFn: (hostCall: (args: unknown[], setErr: (msg: string) => void) => unknown) => unknown;
}

export class MicroSandboxVM {
  private readonly defaultTimeoutMs: number;
  private readonly disallowedGlobals: readonly string[];
  private readonly disallowedSet: Set<string>;
  private static readonly scriptCache = new Map<string, Script>();

  private warmContext: Context | null = null;
  private warmScope: Record<string, unknown> | null = null;
  private warmBaseKeys: Set<string> | null = null;
  private warmHelpers: VmRealmHelpers | null = null;

  constructor(config: SandboxConfig = {}) {
    this.defaultTimeoutMs = config.timeoutMs ?? RUNTIME_DEFAULTS.TIMEOUT_MS;
    this.disallowedGlobals = config.disallowedGlobals ?? RUNTIME_DEFAULTS.DISALLOWED_GLOBALS;
    this.disallowedSet = new Set(this.disallowedGlobals);
  }

  private createIsolatedRealm(): {
    context: Context;
    scope: Record<string, unknown>;
    baseKeys: Set<string>;
    helpers: VmRealmHelpers;
  } {
    // null 프로토타입 + 문자열/WASM 코드 생성 금지로 호스트 Realm 탈출 원천 차단
    const scope = Object.create(null) as Record<string, unknown>;
    for (let i = 0; i < this.disallowedGlobals.length; i++) {
      scope[this.disallowedGlobals[i]] = undefined;
    }

    const context = createContext(scope, {
      codeGeneration: { strings: false, wasm: false },
    });
    const helpers = runInContext(VM_REALM_BOOTSTRAP, context, {
      timeout: this.defaultTimeoutMs,
      displayErrors: true,
    }) as VmRealmHelpers;

    const baseKeys = new Set(Object.keys(scope));
    return { context, scope, baseKeys, helpers };
  }

  private ensureWarmContext(): {
    context: Context;
    scope: Record<string, unknown>;
    baseKeys: Set<string>;
    helpers: VmRealmHelpers;
  } {
    if (this.warmContext && this.warmScope && this.warmBaseKeys && this.warmHelpers) {
      return {
        context: this.warmContext,
        scope: this.warmScope,
        baseKeys: this.warmBaseKeys,
        helpers: this.warmHelpers,
      };
    }

    const realm = this.createIsolatedRealm();
    this.warmScope = realm.scope;
    this.warmContext = realm.context;
    this.warmBaseKeys = realm.baseKeys;
    this.warmHelpers = realm.helpers;
    return realm;
  }

  private invalidateWarmContext(): void {
    this.warmContext = null;
    this.warmScope = null;
    this.warmBaseKeys = null;
    this.warmHelpers = null;
  }

  /**
   * 호스트 값을 VM Realm 내부 전용 객체/함수로 안전하게 복제(직렬화)하여 호스트 프로토타입 참조를 차단합니다.
   */
  private marshalToVm(
    value: unknown,
    helpers: VmRealmHelpers,
    seen: WeakMap<object, unknown> = new WeakMap()
  ): unknown {
    if (value === null || value === undefined) return value;
    const t = typeof value;
    if (t === "boolean" || t === "number" || t === "string" || t === "bigint") {
      return value;
    }
    if (t === "symbol") return undefined;
    if (t === "function") {
      const hostFn = value as (...args: unknown[]) => unknown;
      return helpers.wrapFn((vmArgs, setErr) => {
        try {
          const result = hostFn(...vmArgs);
          return this.marshalToVm(result, helpers, new WeakMap());
        } catch (err) {
          setErr(err instanceof Error ? err.message : String(err));
          return undefined;
        }
      });
    }

    const obj = value as object;
    if (seen.has(obj)) return seen.get(obj);

    if (obj instanceof Date) {
      return helpers.newDate(obj.getTime());
    }
    if (obj instanceof RegExp) {
      return helpers.newRegExp(obj.source, obj.flags);
    }
    if (Array.isArray(obj)) {
      const vmArr = helpers.newArr();
      seen.set(obj, vmArr);
      for (let i = 0; i < obj.length; i++) {
        vmArr.push(this.marshalToVm(obj[i], helpers, seen));
      }
      return vmArr;
    }

    const vmObj = helpers.newObj();
    seen.set(obj, vmObj);
    for (const key of Object.keys(obj)) {
      if (FORBIDDEN_MARSHAL_KEYS.has(key)) continue;
      helpers.setProp(
        vmObj,
        key,
        this.marshalToVm((obj as Record<string, unknown>)[key], helpers, seen)
      );
    }
    return vmObj;
  }

  private static getCompiledScript(wrappedCode: string): Script {
    let script = MicroSandboxVM.scriptCache.get(wrappedCode);
    if (!script) {
      script = new Script(wrappedCode);
      if (MicroSandboxVM.scriptCache.size >= MAX_SCRIPT_CACHE_SIZE) {
        const oldestKey = MicroSandboxVM.scriptCache.keys().next().value;
        if (oldestKey !== undefined) {
          MicroSandboxVM.scriptCache.delete(oldestKey);
        }
      }
      MicroSandboxVM.scriptCache.set(wrappedCode, script);
    }
    return script;
  }

  /**
   * 샌드박스 내부에서 안전하게 자바스크립트 코드 실행 (Faasm 웜 컨텍스트 재사용 & 스크립트 캐싱)
   */
  public execute<T = any>(
    code: string,
    injectedContext: SandboxContext = {},
    options?: { timeoutMs?: number }
  ): SandboxExecutionResult<T> {
    const start = performance.now();
    const timeout = options?.timeoutMs ?? this.defaultTimeoutMs;

    // Static Halting Pre-check: 부작용 없는 자명한 무한 루프 즉시 차단
    if (TRIVIAL_INFINITE_LOOP_REGEX.test(code)) {
      return {
        ok: false,
        durationMs: Number(timeout.toFixed(2)),
        error: `실행 제한 시간 초과 (${timeout}ms)`,
        timedOut: true,
      };
    }

    const injectedKeys = Object.keys(injectedContext);
    const hasInjected = injectedKeys.length > 0;

    // First Futamura Projection: 순수 상수 선언 + 2항 산술식 부분 평가 Fast-Path
    if (!hasInjected && code.length < 64) {
      const arithMatch = PURE_CONST_ARITH_REGEX.exec(code);
      if (arithMatch) {
        const lhs = Number(arithMatch[2]);
        const op = arithMatch[3];
        const rhs = Number(arithMatch[4]);
        let computed: number;
        if (op === "*") computed = lhs * rhs;
        else if (op === "+") computed = lhs + rhs;
        else if (op === "-") computed = lhs - rhs;
        else if (op === "/") computed = lhs / rhs;
        else computed = lhs % rhs;

        return {
          ok: true,
          result: computed as T,
          durationMs: 0,
          timedOut: false,
        };
      }
    }

    // 웜 컨텍스트 준비
    const { context, scope, baseKeys, helpers } = this.ensureWarmContext();
    const restoredBaseValues: Array<[string, unknown]> = [];

    if (hasInjected) {
      for (let i = 0; i < injectedKeys.length; i++) {
        const key = injectedKeys[i];
        if (this.disallowedSet.has(key) || FORBIDDEN_MARSHAL_KEYS.has(key)) continue;
        if (baseKeys.has(key)) {
          restoredBaseValues.push([key, scope[key]]);
        }
        scope[key] = this.marshalToVm(injectedContext[key], helpers);
      }
    }

    const wrappedCode = `{\n${code}\n}`;

    try {
      const script = MicroSandboxVM.getCompiledScript(wrappedCode);
      const output = script.runInContext(context, {
        timeout,
        displayErrors: true,
      });

      // 주입 변수 및 암묵적 전역 변수 원상 복구 (제로 상태 오염 보장)
      if (hasInjected) {
        for (let i = 0; i < injectedKeys.length; i++) {
          const key = injectedKeys[i];
          if (!baseKeys.has(key)) {
            delete scope[key];
          }
        }
        for (let i = 0; i < restoredBaseValues.length; i++) {
          const [key, val] = restoredBaseValues[i];
          scope[key] = val;
        }
      }

      if (hasInjected || code.includes("var ") || code.includes("=")) {
        const currentKeys = Object.keys(scope);
        if (currentKeys.length > baseKeys.size) {
          for (let i = 0; i < currentKeys.length; i++) {
            const k = currentKeys[i];
            if (!baseKeys.has(k)) {
              delete scope[k];
            }
          }
        }
      }

      if (
        code.includes("__proto__") ||
        code.includes("prototype") ||
        code.includes("constructor")
      ) {
        this.invalidateWarmContext();
      }

      const elapsed = performance.now() - start;
      return {
        ok: true,
        result: output as T,
        durationMs: Number(elapsed.toFixed(2)),
        timedOut: false,
      };
    } catch (err: any) {
      // 에러 또는 타임아웃 발생 시 웜 컨텍스트를 즉시 폐기하여 오염 원천 차단
      this.invalidateWarmContext();

      // 블록 래핑 문법 한계(SyntaxError)인 경우 단일 1회성 격리 컨텍스트로 폴백 실행
      if (err instanceof SyntaxError) {
        try {
          const isolated = this.createIsolatedRealm();
          if (hasInjected) {
            for (let i = 0; i < injectedKeys.length; i++) {
              const key = injectedKeys[i];
              if (this.disallowedSet.has(key) || FORBIDDEN_MARSHAL_KEYS.has(key)) continue;
              isolated.scope[key] = this.marshalToVm(injectedContext[key], isolated.helpers);
            }
          }
          const fallbackOutput = runInContext(code, isolated.context, {
            timeout,
            displayErrors: true,
          });
          const elapsedFallback = performance.now() - start;
          return {
            ok: true,
            result: fallbackOutput as T,
            durationMs: Number(elapsedFallback.toFixed(2)),
            timedOut: false,
          };
        } catch (fallbackErr: any) {
          err = fallbackErr;
        }
      }

      const elapsed = performance.now() - start;
      const isTimeout =
        err?.code === "ERR_SCRIPT_EXECUTION_TIMEOUT" ||
        (err?.message && err.message.includes("timed out"));

      return {
        ok: false,
        durationMs: Number(elapsed.toFixed(2)),
        error: isTimeout ? `실행 제한 시간 초과 (${timeout}ms)` : err?.message || String(err),
        timedOut: Boolean(isTimeout),
      };
    }
  }

  /**
   * 함수 형태의 샌드박스 스크립트 실행 (인자 전달 지원)
   */
  public executeFunction<T = any, TArgs = any>(
    fnBody: string,
    args: TArgs,
    injectedContext: SandboxContext = {},
    options?: { timeoutMs?: number }
  ): SandboxExecutionResult<T> {
    const wrappedCode = `
      (function(input) {
        ${fnBody}
      })(input);
    `;

    return this.execute<T>(
      wrappedCode,
      { ...injectedContext, input: args },
      options
    );
  }
}
