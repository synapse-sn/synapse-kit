/**
 * WebAssembly v1 네이티브 바이너리 에미터 (Zero-Dependency WASM Binary Emitter)
 * - 시냅스 언어의 `native fn ... via wasm { return <expr>; }` AST를
 *   표준 WASM 바이트코드(\0asm v1)로 직접 컴파일하여 JIT/AOT 기계어 실행을 지원합니다.
 * - 혼합 i32/i64 승격, 단락 평가(&&, ||), 0 나눗셈/INT_MIN 오버플로우 트랩 방어를 보장합니다.
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

type WasmScalarType = "i32" | "i64";

interface WasmEmitContext {
  readonly paramMap: Map<string, { index: number; type: WasmScalarType }>;
  readonly tmpI32A: number;
  readonly tmpI32B: number;
  readonly tmpI64A: number;
  readonly tmpI64B: number;
}

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

const COMPARISON_OPS = new Set(["==", "!=", "<", ">", "<=", ">="]);

function inferExprType(expr: Expr, ctx: WasmEmitContext, hint: WasmScalarType): WasmScalarType {
  switch (expr.kind) {
    case "Identifier": {
      const entry = ctx.paramMap.get(expr.name);
      return entry ? entry.type : hint;
    }
    case "BoolLiteral":
      return "i32";
    case "NumberLiteral":
      return hint;
    case "Unary":
      return expr.op === "!" ? "i32" : inferExprType(expr.operand, ctx, hint);
    case "Binary": {
      if (expr.op === "&&" || expr.op === "||" || COMPARISON_OPS.has(expr.op)) {
        return "i32";
      }
      const lt = inferExprType(expr.left, ctx, "i32");
      const rt = inferExprType(expr.right, ctx, "i32");
      return lt === "i64" || rt === "i64" ? "i64" : hint;
    }
    default:
      return hint;
  }
}

function emitBoolNormalize(actualType: WasmScalarType, out: number[]): void {
  if (actualType === "i64") {
    out.push(0x42, 0x00, 0x52); // i64.const 0; i64.ne -> i32 (0 or 1)
  } else {
    out.push(0x41, 0x00, 0x47); // i32.const 0; i32.ne -> i32 (0 or 1)
  }
}

function emitCoerce(from: WasmScalarType, to: WasmScalarType, out: number[]): void {
  if (from === to) return;
  if (from === "i32" && to === "i64") {
    out.push(0xac); // i64.extend_i32_s
  } else if (from === "i64" && to === "i32") {
    out.push(0xa7); // i32.wrap_i64
  }
}

function emitExpr(
  expr: Expr,
  ctx: WasmEmitContext,
  hint: WasmScalarType,
  out: number[],
): WasmScalarType {
  switch (expr.kind) {
    case "NumberLiteral": {
      if (hint === "i64") {
        out.push(0x42, ...encodeS64(BigInt(Math.trunc(expr.value))));
        return "i64";
      }
      out.push(0x41, ...encodeS32(expr.value | 0));
      return "i32";
    }
    case "BoolLiteral": {
      out.push(0x41, ...encodeS32(expr.value ? 1 : 0));
      return "i32";
    }
    case "Identifier": {
      const entry = ctx.paramMap.get(expr.name);
      if (entry === undefined) {
        throw new SnError(
          SN_MESSAGES.UNSUPPORTED_WASM_EXPR(`알 수 없는 식별자 '${expr.name}'`),
          "WASM_UNKNOWN_IDENT",
          expr.span,
        );
      }
      out.push(0x20, ...encodeU32(entry.index)); // local.get
      return entry.type;
    }
    case "Unary": {
      if (expr.op === "-") {
        const innerType = inferExprType(expr.operand, ctx, hint);
        if (innerType === "i64") {
          out.push(0x42, 0x00);
          const actual = emitExpr(expr.operand, ctx, "i64", out);
          emitCoerce(actual, "i64", out);
          out.push(0x7d); // i64.sub
          return "i64";
        }
        out.push(0x41, 0x00);
        const actual = emitExpr(expr.operand, ctx, "i32", out);
        emitCoerce(actual, "i32", out);
        out.push(0x6b); // i32.sub
        return "i32";
      }
      if (expr.op === "!") {
        const actual = emitExpr(expr.operand, ctx, "i32", out);
        out.push(actual === "i64" ? 0x50 : 0x45); // i64.eqz / i32.eqz -> always i32
        return "i32";
      }
      throw new SnError(
        SN_MESSAGES.UNSUPPORTED_WASM_EXPR(`지원하지 않는 단항 연산자 '${expr.op}'`),
        "WASM_UNSUPPORTED_OP",
        expr.span,
      );
    }
    case "Binary": {
      // 논리 연산 && / || : WASM control-flow if/else로 진정한 단락 평가 및 0/1 불리언 정규화
      if (expr.op === "&&") {
        const lt = emitExpr(expr.left, ctx, "i32", out);
        emitBoolNormalize(lt, out);
        out.push(0x04, VALTYPE_I32); // if (result i32)
        const rt = emitExpr(expr.right, ctx, "i32", out);
        emitBoolNormalize(rt, out);
        out.push(0x05, 0x41, 0x00, 0x0b); // else i32.const 0; end
        return "i32";
      }
      if (expr.op === "||") {
        const lt = emitExpr(expr.left, ctx, "i32", out);
        emitBoolNormalize(lt, out);
        out.push(0x04, VALTYPE_I32); // if (result i32)
        out.push(0x41, 0x01); // i32.const 1
        out.push(0x05); // else
        const rt = emitExpr(expr.right, ctx, "i32", out);
        emitBoolNormalize(rt, out);
        out.push(0x0b); // end
        return "i32";
      }

      const inferredL = inferExprType(expr.left, ctx, "i32");
      const inferredR = inferExprType(expr.right, ctx, "i32");
      const operandType: WasmScalarType =
        inferredL === "i64" || inferredR === "i64" || hint === "i64" ? "i64" : "i32";

      // 나눗셈(/, %): 0 나눗셈 및 INT_MIN / -1 트랩 방지 (JS/engine과 동일하게 0 반환)
      if (expr.op === "/" || expr.op === "%") {
        const actualL = emitExpr(expr.left, ctx, operandType, out);
        emitCoerce(actualL, operandType, out);
        const tmpL = operandType === "i64" ? ctx.tmpI64A : ctx.tmpI32A;
        const tmpR = operandType === "i64" ? ctx.tmpI64B : ctx.tmpI32B;
        out.push(0x21, ...encodeU32(tmpL)); // local.set tmpL

        const actualR = emitExpr(expr.right, ctx, operandType, out);
        emitCoerce(actualR, operandType, out);
        out.push(0x21, ...encodeU32(tmpR)); // local.set tmpR

        if (operandType === "i64") {
          // if (tmpR == 0 || (tmpL == i64.MIN && tmpR == -1)) -> 0 else div_s/rem_s
          out.push(0x20, ...encodeU32(tmpR), 0x50); // tmpR == 0
          out.push(0x20, ...encodeU32(tmpL), 0x42, ...encodeS64(-9223372036854775808n), 0x51);
          out.push(0x20, ...encodeU32(tmpR), 0x42, ...encodeS64(-1n), 0x51);
          out.push(0x71, 0x72); // i32.and, i32.or
          out.push(0x04, VALTYPE_I64, 0x42, 0x00, 0x05);
          out.push(0x20, ...encodeU32(tmpL), 0x20, ...encodeU32(tmpR), expr.op === "/" ? 0x7f : 0x81);
          out.push(0x0b);
          return "i64";
        }

        out.push(0x20, ...encodeU32(tmpR), 0x45); // tmpR == 0
        out.push(0x20, ...encodeU32(tmpL), 0x41, ...encodeS32(-2147483648), 0x46);
        out.push(0x20, ...encodeU32(tmpR), 0x41, ...encodeS32(-1), 0x46);
        out.push(0x71, 0x72); // i32.and, i32.or
        out.push(0x04, VALTYPE_I32, 0x41, 0x00, 0x05);
        out.push(0x20, ...encodeU32(tmpL), 0x20, ...encodeU32(tmpR), expr.op === "/" ? 0x6d : 0x6f);
        out.push(0x0b);
        return "i32";
      }

      const actualL = emitExpr(expr.left, ctx, operandType, out);
      emitCoerce(actualL, operandType, out);
      const actualR = emitExpr(expr.right, ctx, operandType, out);
      emitCoerce(actualR, operandType, out);

      const opMapI32: Record<string, number> = {
        "+": 0x6a,
        "-": 0x6b,
        "*": 0x6c,
        "==": 0x46,
        "!=": 0x47,
        "<": 0x48,
        ">": 0x4a,
        "<=": 0x4c,
        ">=": 0x4e,
      };
      const opMapI64: Record<string, number> = {
        "+": 0x7c,
        "-": 0x7d,
        "*": 0x7e,
        "==": 0x51,
        "!=": 0x52,
        "<": 0x53,
        ">": 0x55,
        "<=": 0x57,
        ">=": 0x59,
      };
      const opcode = (operandType === "i64" ? opMapI64 : opMapI32)[expr.op];
      if (opcode === undefined) {
        throw new SnError(
          SN_MESSAGES.UNSUPPORTED_WASM_EXPR(`지원하지 않는 이항 연산자 '${expr.op}'`),
          "WASM_UNSUPPORTED_OP",
          expr.span,
        );
      }
      out.push(opcode);
      return COMPARISON_OPS.has(expr.op) ? "i32" : operandType;
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

  const paramMap = new Map<string, { index: number; type: WasmScalarType }>();
  fn.params.forEach((p, idx) =>
    paramMap.set(p.name, { index: idx, type: p.type === "i64" ? "i64" : "i32" }),
  );
  const paramCount = fn.params.length;
  const ctx: WasmEmitContext = {
    paramMap,
    tmpI32A: paramCount,
    tmpI32B: paramCount + 1,
    tmpI64A: paramCount + 2,
    tmpI64B: paramCount + 3,
  };

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
  const targetRetType: WasmScalarType = fn.returnType === "i64" ? "i64" : "i32";
  const bodyInstr: number[] = [];
  const bodyType = emitExpr(fn.bodyExpr, ctx, targetRetType, bodyInstr);
  if (fn.returnType === "bool") {
    emitBoolNormalize(bodyType, bodyInstr);
  } else {
    emitCoerce(bodyType, targetRetType, bodyInstr);
  }
  bodyInstr.push(0x0b); // end

  // 스크래치 로컬 2개(i32) + 2개(i64) 선언
  const funcBody = [0x02, 0x02, VALTYPE_I32, 0x02, VALTYPE_I64, ...bodyInstr];
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
