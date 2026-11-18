/**
 * 시냅스 제어 언어 통합 컴파일러 & 코드 생성기 (Compiler & Multi-Target Codegen)
 * - AST -> 정적 검증 -> WASM 바이너리 / Rust C ABI FFI 스텁 / TypeScript RingBufferBus 코드 생성
 */
import { createHash } from "node:crypto";
import { analyzeProgram } from "./analyzer";
import { SN_DEFAULTS } from "./config";
import { parseSn } from "./parser";
import {
  SnError,
  type CompiledSnUnit,
  type CompiledWasmFunction,
  type NativeFnDecl,
  type PrimitiveType,
  type SnProgram,
} from "./types";
import { compileNativeFnToWasm } from "./wasmEmitter";

function toRustType(t: PrimitiveType): string {
  if (t === "i64") return "i64";
  if (t === "bool") return "bool";
  return "i32";
}

function generateRustFfiStub(program: SnProgram): string {
  const lines: string[] = [
    "// 자동 생성된 Synapse Control Plane Rust C ABI 스텁",
    `// IR Version: ${SN_DEFAULTS.IR_VERSION}`,
    "#![allow(dead_code)]",
    "",
  ];

  const ffiFns = program.nativeFns.filter((f) => f.transport === "ffi");
  if (ffiFns.length > 0) {
    lines.push('extern "C" {');
    for (const fn of ffiFns) {
      const params = fn.params.map((p) => `${p.name}: ${toRustType(p.type)}`).join(", ");
      lines.push(`    pub fn ${fn.name}(${params}) -> ${toRustType(fn.returnType)};`);
    }
    lines.push("}");
    lines.push("");
  }

  for (const wire of program.wires) {
    lines.push(
      `pub const WIRE_${wire.name.toUpperCase()}_FROM: &str = ${JSON.stringify(wire.fromTopic)};`,
    );
    lines.push(
      `pub const WIRE_${wire.name.toUpperCase()}_TO: &str = ${JSON.stringify(wire.toTarget)};`,
    );
  }

  return lines.join("\n");
}

function generateTsBridgeCode(program: SnProgram): string {
  const lines: string[] = [
    "// 자동 생성된 Synapse Control Plane TypeScript(RingBufferBus) 브리지",
    `// IR Version: ${SN_DEFAULTS.IR_VERSION}`,
    'import type { RingBufferBus } from "../bus";',
    "",
    "export function registerGeneratedWires(bus: RingBufferBus): void {",
  ];

  for (const wire of program.wires) {
    lines.push(`  // 와이어: ${wire.name} (오염등급: ${wire.sourceTaint})`);
    lines.push(`  bus.subscribe(${JSON.stringify(wire.fromTopic)}, (msg) => {`);
    lines.push(`    bus.emit(${JSON.stringify(wire.toTarget)}, msg.payload, ${JSON.stringify(wire.name)});`);
    lines.push("  });");
  }

  lines.push("}");
  return lines.join("\n");
}

export interface CompileOptions {
  /** 에러 진단 발견 시 즉시 예외를 던질지 여부 (기본값: true) */
  strict?: boolean;
}

export function compileSn(
  source: string,
  options: CompileOptions = {},
): CompiledSnUnit {
  const strict = options.strict ?? true;
  const program = parseSn(source);
  const diagnostics = analyzeProgram(program);

  const errors = diagnostics.filter((d) => d.severity === "error");
  if (strict && errors.length > 0) {
    throw new SnError(
      errors.map((e) => e.message).join("\n"),
      errors[0].code,
      errors[0].span,
    );
  }

  const wasmFunctions = new Map<string, CompiledWasmFunction>();
  for (const fn of program.nativeFns) {
    if (fn.transport === "wasm" && fn.bodyExpr) {
      const compiled = compileNativeFnToWasm(fn);
      wasmFunctions.set(fn.name, compiled);
    }
  }

  const canonical = JSON.stringify({
    irVersion: SN_DEFAULTS.IR_VERSION,
    program,
  });
  const irHash = createHash("sha256").update(canonical).digest("hex");

  return {
    irVersion: SN_DEFAULTS.IR_VERSION,
    irHash,
    program,
    diagnostics,
    wasmFunctions,
    rustFfiStub: generateRustFfiStub(program),
    tsBridgeCode: generateTsBridgeCode(program),
  };
}

export function describeNativeFn(fn: NativeFnDecl): string {
  const params = fn.params.map((p) => `${p.name}: ${p.type}`).join(", ");
  return `native fn ${fn.name}(${params}) -> ${fn.returnType} via ${fn.transport}`;
}
