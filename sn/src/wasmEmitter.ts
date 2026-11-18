/**
 * WebAssembly v1 네이티브 바이너리 에미터 (Zero-Dependency WASM Binary Emitter)
 * - 시냅스 언어의 `native fn ... via wasm { return <expr>; }` AST를
 *   표준 WASM 바이트코드(\0asm v1)로 직접 컴파일하여 JIT/AOT 기계어 실행을 지원합니다.
 */
import { SN_MESSAGES } from "./config";
import {
  SnError,
  type CompiledWasmFunction,
  type Expr,
  type NativeFnDecl,
  type PrimitiveType,
} from "./types";

const WASM_MAGIC = [0x00, 0x61, 0x73, 0x6d];
const WASM_VERSION = [0x01, 0x00, 0x00, 0x00];

const VALTYPE_I32 = 0x7f;
const VALTYPE_I64 = 0x7e;

function toValType(t: PrimitiveType): number {
  return t === "i64" ? VALTYPE_I64 : VALTYPE_I32;
}

function encodeU32(value: number): number[] {
  const out: number[] = [];
  let v = value >>> 0;
  do {
    let byte = v & 0x7f;
    v >>>= 7;
    if (v !== 0) byte |= 0x80;
    out.push(byte);
  } while (v !== 0);
  return out;
}

function encodeS32(value: number): number[] {
  const out: number[] = [];
  let v = value | 0;
  let more = true;
  while (more) {
    let byte = v & 0x7f;
    v >>= 7;
    if ((v === 0 && (byte & 0x40) === 0) || (v === -1 && (byte & 0x40) !== 0)) {
      more = false;
    } else {
      byte |= 0x80;
    }
    out.push(byte);
  }
  return out;
}

function encodeS64(value: bigint): number[] {
  const out: number[] = [];
  let v = value;
  let more = true;
  while (more) {
    let byte = Number(v & 0x7fn);
    v >>= 7n;
    if ((v === 0n && (byte & 0x40) === 0) || (v === -1n && (byte & 0x40) !== 0)) {
      more = false;
    } else {
      byte |= 0x80;
    }
    out.push(byte);
  }
  return out;
}

function encodeSection(sectionId: number, payload: number[]): number[] {
  return [sectionId, ...encodeU32(payload.length), ...payload];
}

function encodeString(str: string): number[] {
  const bytes = Array.from(new TextEncoder().encode(str));
  return [...encodeU32(bytes.length), ...bytes];
}

function emitExpr(
  expr: Expr,
  paramMap: Map<string, number>,
  isI64: boolean,
  out: number[],
): void {
  switch (expr.kind) {
    case "NumberLiteral": {
      if (isI64) {
        out.push(0x42, ...encodeS64(BigInt(Math.trunc(expr.value)))); // i64.const
      } else {
        out.push(0x41, ...encodeS32(expr.value | 0)); // i32.const
      }
      return;
    }
    case "BoolLiteral": {
      if (isI64) {
        out.push(0x42, ...encodeS64(expr.value ? 1n : 0n));
      } else {
        out.push(0x41, ...encodeS32(expr.value ? 1 : 0));
      }
      return;
    }
    case "Identifier": {
      const idx = paramMap.get(expr.name);
      if (idx === undefined) {
        throw new SnError(
          SN_MESSAGES.UNSUPPORTED_WASM_EXPR(`알 수 없는 식별자 '${expr.name}'`),
          "WASM_UNKNOWN_IDENT",
          expr.span,
        );
      }
      out.push(0x20, ...encodeU32(idx)); // local.get
      return;
    }
    case "Unary": {
      if (expr.op === "-") {
        if (isI64) {
          out.push(0x42, 0x00);
          emitExpr(expr.operand, paramMap, true, out);
          out.push(0x7d); // i64.sub
        } else {
          out.push(0x41, 0x00);
          emitExpr(expr.operand, paramMap, false, out);
          out.push(0x6b); // i32.sub
        }
        return;
      }
      if (expr.op === "!") {
        emitExpr(expr.operand, paramMap, isI64, out);
        out.push(isI64 ? 0x50 : 0x45); // i64.eqz / i32.eqz
        return;
      }
      throw new SnError(
        SN_MESSAGES.UNSUPPORTED_WASM_EXPR(`지원하지 않는 단항 연산자 '${expr.op}'`),
        "WASM_UNSUPPORTED_OP",
        expr.span,
      );
    }
    case "Binary": {
      emitExpr(expr.left, paramMap, isI64, out);
      emitExpr(expr.right, paramMap, isI64, out);
      const opMapI32: Record<string, number> = {
        "+": 0x6a,
        "-": 0x6b,
        "*": 0x6c,
        "/": 0x6d,
        "%": 0x6f,
        "==": 0x46,
        "!=": 0x47,
        "<": 0x48,
        ">": 0x4a,
        "<=": 0x4c,
        ">=": 0x4e,
        "&&": 0x71,
        "||": 0x72,
      };
      const opMapI64: Record<string, number> = {
        "+": 0x7c,
        "-": 0x7d,
        "*": 0x7e,
        "/": 0x7f,
        "%": 0x81,
        "==": 0x51,
        "!=": 0x52,
        "<": 0x53,
        ">": 0x55,
        "<=": 0x57,
        ">=": 0x59,
        "&&": 0x83,
        "||": 0x84,
      };
      const opcode = (isI64 ? opMapI64 : opMapI32)[expr.op];
      if (opcode === undefined) {
        throw new SnError(
          SN_MESSAGES.UNSUPPORTED_WASM_EXPR(`지원하지 않는 이항 연산자 '${expr.op}'`),
          "WASM_UNSUPPORTED_OP",
          expr.span,
        );
      }
      out.push(opcode);
      return;
    }
    default:
      throw new SnError(
        SN_MESSAGES.UNSUPPORTED_WASM_EXPR(`지원하지 않는 표현식 '${expr.kind}'`),
        "WASM_UNSUPPORTED_EXPR",
        expr.span,
      );
  }
}

export function compileNativeFnToWasm(fn: NativeFnDecl): CompiledWasmFunction {
  if (!fn.bodyExpr) {
    throw new SnError(
      SN_MESSAGES.UNSUPPORTED_WASM_EXPR(`함수 '${fn.name}'에 본문이 없습니다.`),
      "WASM_MISSING_BODY",
      fn.span,
    );
  }

  const paramMap = new Map<string, number>();
  fn.params.forEach((p, idx) => paramMap.set(p.name, idx));
  const isI64 = fn.returnType === "i64" || fn.params.some((p) => p.type === "i64");

  // 1. Type section (0x01): 1 func signature
  const paramTypes = fn.params.map((p) => toValType(p.type));
  const resultTypes = [toValType(fn.returnType)];
  const typeSection = encodeSection(0x01, [
    0x01, // count = 1
    0x60, // functype
    ...encodeU32(paramTypes.length),
    ...paramTypes,
    ...encodeU32(resultTypes.length),
    ...resultTypes,
  ]);

  // 2. Function section (0x03): 1 function using type 0
  const funcSection = encodeSection(0x03, [0x01, 0x00]);

  // 3. Export section (0x07): export fn.name -> func 0
  const exportSection = encodeSection(0x07, [
    0x01,
    ...encodeString(fn.name),
    0x00, // kind = func
    0x00, // index = 0
  ]);

  // 4. Code section (0x0a)
  const bodyInstr: number[] = [];
  emitExpr(fn.bodyExpr, paramMap, isI64, bodyInstr);
  bodyInstr.push(0x0b); // end

  const funcBody = [0x00, ...bodyInstr]; // 0 local declarations
  const codeSection = encodeSection(0x0a, [
    0x01, // 1 function body
    ...encodeU32(funcBody.length),
    ...funcBody,
  ]);

  const bytes = new Uint8Array([
    ...WASM_MAGIC,
    ...WASM_VERSION,
    ...typeSection,
    ...funcSection,
    ...exportSection,
    ...codeSection,
  ]);

  if (!WebAssembly.validate(bytes)) {
    throw new SnError(
      SN_MESSAGES.UNSUPPORTED_WASM_EXPR(`생성된 WASM 바이너리('${fn.name}')가 표준 검증을 통과하지 못했습니다.`),
      "WASM_VALIDATION_FAILED",
      fn.span,
    );
  }

  return {
    name: fn.name,
    params: fn.params,
    returnType: fn.returnType,
    wasmBytes: bytes,
  };
}
