import { describe, expect, test } from "bun:test";
import { dlopen, FFIType, suffix } from "bun:ffi";
import { existsSync } from "node:fs";
import { resolve } from "node:path";
import { RingBufferBus } from "../../bus/src/ringBufferBus";
import {
  SnEngine,
  SnError,
  compileSn,
} from "../src";

const SAMPLE_CONTROL_SCRIPT = `
// 시냅스 제어 플레인 표준 시나리오
policy strict_edge {
  taint untrusted -> deny shell, fs_write, native_ffi;
  allow bus_emit, vault_read;
  quota timeout_ms = 200;
  quota max_memory_mb = 32;
}

native fn fast_score(a: i32, b: i32) -> i32 via wasm {
  return (a * 2) + b;
}

native fn is_within_limit(val: i32, limit: i32) -> bool via wasm {
  return val <= limit && val >= 0;
}

native fn rust_blake3(ptr: i64, len: i32) -> i64 via ffi "synapse_native_core";

invariant clip_payload_valid {
  cause "클립 점수가 음수이거나 허용 상한을 초과했습니다.";
  require is_within_limit(payload.score, 100) && payload.len > 0;
  effect "전송 차단 및 dead_letter 폴백 라우팅";
  severity critical;
}

wire aside_to_bot {
  from "aside.clip.captured" as untrusted;
  guard strict_edge;
  sanitize clip_sanitizer;
  check clip_payload_valid;
  to "bot.discord.publish";
  on_error "runner.dead_letter";
}
`;

describe("🧠 Synapse Control Language (synapse-lang) 코어 검증", () => {
  test("1. [파싱 & IR 해시] policy/native/invariant/wire 선언을 결정론적으로 컴파일한다", () => {
    const unit1 = compileSn(SAMPLE_CONTROL_SCRIPT);
    const unit2 = compileSn(SAMPLE_CONTROL_SCRIPT);

    expect(unit1.irHash).toBe(unit2.irHash);
    expect(unit1.program.policies).toHaveLength(1);
    expect(unit1.program.nativeFns).toHaveLength(3);
    expect(unit1.program.invariants).toHaveLength(1);
    expect(unit1.program.wires).toHaveLength(1);
    expect(unit1.diagnostics).toHaveLength(0);
  });

  test("2. [CaMeL Taint 차단] untrusted 소스가 sanitize 없이 민감 싱크로 흐르면 컴파일 에러로 차단한다", () => {
    const unsafeScript = `
      policy edge_guard {
        taint untrusted -> deny shell, native_ffi;
      }

      wire dangerous_wire {
        from "web.webhook.raw" as untrusted;
        guard edge_guard;
        to "shell:exec";
      }
    `;

    expect(() => compileSn(unsafeScript)).toThrow(SnError);
    const nonStrict = compileSn(unsafeScript, { strict: false });
    expect(nonStrict.diagnostics.some((d) => d.code === "TAINT_VIOLATION")).toBe(true);
  });

  test("3. [네이티브 기계어 WASM 컴파일 & JIT 실행] via wasm 함수가 표준 WASM 바이너리로 컴파일되어 실행된다", async () => {
    const engine = new SnEngine(SAMPLE_CONTROL_SCRIPT);
    await engine.init();

    const wasmFn = engine.getCompiledUnit().wasmFunctions.get("fast_score");
    expect(wasmFn).toBeDefined();
    expect(Array.from(wasmFn!.wasmBytes.slice(0, 4))).toEqual([0x00, 0x61, 0x73, 0x6d]);
    expect(WebAssembly.validate(wasmFn!.wasmBytes)).toBe(true);

    expect(engine.callNative("fast_score", 15, 7)).toBe(37);
    expect(engine.callNative("is_within_limit", 50, 100)).toBe(1);
    expect(engine.callNative("is_within_limit", 150, 100)).toBe(0);
  });

  test("4. [Rust C ABI FFI 스텁 생성] via ffi 선언으로부터 extern \"C\" 바인딩을 생성한다", () => {
    const unit = compileSn(SAMPLE_CONTROL_SCRIPT);
    expect(unit.rustFfiStub).toContain('extern "C"');
    expect(unit.rustFfiStub).toContain("pub fn rust_blake3(ptr: i64, len: i32) -> i64;");
    expect(unit.rustFfiStub).toContain('WIRE_ASIDE_TO_BOT_FROM: &str = "aside.clip.captured"');
  });

  test("5. [RingBufferBus 실전 제어] 와이어 바인딩 후 정상 페이로드 통과 및 불변식 위반 시 dead_letter 라우팅", async () => {
    const bus = new RingBufferBus({ ringCapacity: 64 });
    const engine = new SnEngine(SAMPLE_CONTROL_SCRIPT);
    await engine.init();

    engine.registerSanitizer("clip_sanitizer", (raw: any) => ({
      ...raw,
      sanitized: true,
      title: String(raw?.title ?? "").replace(/<script>/gi, ""),
    }));

    const published: any[] = [];
    const deadLetters: any[] = [];
    bus.subscribe("bot.discord.publish", (msg) => published.push(msg.payload));
    bus.subscribe("runner.dead_letter", (msg) => deadLetters.push(msg.payload));

    engine.bindBus(bus);

    // 정상 페이로드
    bus.emit("aside.clip.captured", { title: "<script>하이라이트", score: 85, len: 12 });
    expect(published).toHaveLength(1);
    expect(published[0].sanitized).toBe(true);
    expect(published[0].title).toBe("하이라이트");

    // 불변식 위반 페이로드 (score > 100)
    bus.emit("aside.clip.captured", { title: "비정상", score: 999, len: 5 });
    expect(published).toHaveLength(1);
    expect(deadLetters).toHaveLength(1);
    expect(deadLetters[0].wire).toBe("aside_to_bot");
    expect(deadLetters[0].reason).toContain("clip_payload_valid");

    const audit = engine.getAuditLog();
    expect(audit.map((a) => a.status)).toEqual(["delivered", "blocked_invariant"]);

    engine.unbindBus();
  });

  test("6. [Rust 네이티브 기계어 FFI 연동] sn_native DLL(C ABI)을 bun:ffi로 로드해 sn 불변식에서 직접 호출한다", async () => {
    const dllPath = resolve(import.meta.dir, `../native/target/release/sn_native.${suffix}`);
    expect(existsSync(dllPath)).toBe(true);

    const lib = dlopen(dllPath, {
      sn_score_gate: {
        args: [FFIType.i32, FFIType.i32],
        returns: FFIType.i32,
      },
      sn_weighted_sum: {
        args: [FFIType.i32, FFIType.i32],
        returns: FFIType.i32,
      },
    });

    try {
      const ffiScript = `
        native fn sn_score_gate(score: i32, limit: i32) -> i32 via ffi "sn_native";
        native fn sn_weighted_sum(a: i32, b: i32) -> i32 via ffi "sn_native";

        invariant rust_gate_ok {
          cause "Rust 네이티브 점수 게이트 통과 실패";
          require sn_score_gate(payload.score, 100) == 1;
          severity critical;
        }

        wire rust_checked_wire {
          from "rust.in";
          check rust_gate_ok;
          to "rust.out";
          on_error "rust.dead_letter";
        }
      `;

      const bus = new RingBufferBus({ ringCapacity: 32 });
      const engine = new SnEngine(ffiScript);
      await engine.init();

      engine.registerFfiHandler("sn_score_gate", lib.symbols.sn_score_gate);
      engine.registerFfiHandler("sn_weighted_sum", lib.symbols.sn_weighted_sum);

      // 직접 네이티브 Rust 기계어 호출 검증
      expect(engine.callNative("sn_weighted_sum", 15, 7)).toBe(37);

      const passed: any[] = [];
      const blocked: any[] = [];
      bus.subscribe("rust.out", (m) => passed.push(m.payload));
      bus.subscribe("rust.dead_letter", (m) => blocked.push(m.payload));

      engine.bindBus(bus);
      bus.emit("rust.in", { score: 85 });
      bus.emit("rust.in", { score: 150 });

      expect(passed).toEqual([{ score: 85 }]);
      expect(blocked).toHaveLength(1);
      expect(blocked[0].reason).toContain("rust_gate_ok");

      engine.unbindBus();
    } finally {
      lib.close();
    }
  });
});
