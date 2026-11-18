/**
 * 시냅스 제어 언어 (Synapse Control Language) 핵심 타입 및 AST 정의
 */

export type TaintLevel = "trusted" | "untrusted" | "sanitized";
export type PrimitiveType = "i32" | "i64" | "bool";
export type SeverityLevel = "critical" | "warn";
export type NativeTransport = "wasm" | "ffi" | "bus";

export interface SourceSpan {
  line: number;
  column: number;
}

export type TokenKind =
  | "KEYWORD"
  | "IDENT"
  | "NUMBER"
  | "STRING"
  | "OP"
  | "PUNCT"
  | "EOF";

export interface Token {
  kind: TokenKind;
  value: string;
  line: number;
  column: number;
}

export type Expr =
  | { kind: "NumberLiteral"; value: number; isFloat: boolean; span: SourceSpan }
  | { kind: "BoolLiteral"; value: boolean; span: SourceSpan }
  | { kind: "StringLiteral"; value: string; span: SourceSpan }
  | { kind: "Identifier"; name: string; span: SourceSpan }
  | { kind: "MemberAccess"; object: Expr; property: string; span: SourceSpan }
  | { kind: "Unary"; op: "-" | "!"; operand: Expr; span: SourceSpan }
  | {
      kind: "Binary";
      op:
        | "+"
        | "-"
        | "*"
        | "/"
        | "%"
        | "=="
        | "!="
        | "<"
        | "<="
        | ">"
        | ">="
        | "&&"
        | "||";
      left: Expr;
      right: Expr;
      span: SourceSpan;
    }
  | { kind: "Call"; callee: string; args: Expr[]; span: SourceSpan };

export interface TaintRule {
  sourceTaint: TaintLevel;
  deniedCapabilities: string[];
}

export interface PolicyDecl {
  kind: "PolicyDecl";
  name: string;
  taintRules: TaintRule[];
  allowedCapabilities: string[];
  deniedCapabilities: string[];
  quotas: Record<string, number>;
  span: SourceSpan;
}

export interface NativeParam {
  name: string;
  type: PrimitiveType;
}

export interface NativeFnDecl {
  kind: "NativeFnDecl";
  name: string;
  params: NativeParam[];
  returnType: PrimitiveType;
  transport: NativeTransport;
  targetRef?: string;
  bodyExpr?: Expr;
  span: SourceSpan;
}

export interface InvariantDecl {
  kind: "InvariantDecl";
  name: string;
  cause: string;
  requireExpr: Expr;
  effect: string;
  severity: SeverityLevel;
  span: SourceSpan;
}

export interface WireDecl {
  kind: "WireDecl";
  name: string;
  fromTopic: string;
  sourceTaint: TaintLevel;
  guardPolicy?: string;
  sanitizer?: string;
  checks: string[];
  toTarget: string;
  onErrorTopic?: string;
  span: SourceSpan;
}

export interface SnProgram {
  kind: "SnProgram";
  policies: PolicyDecl[];
  nativeFns: NativeFnDecl[];
  invariants: InvariantDecl[];
  wires: WireDecl[];
}

export interface Diagnostic {
  code: string;
  severity: "error" | "warning";
  message: string;
  span?: SourceSpan;
}

export interface CompiledWasmFunction {
  name: string;
  params: NativeParam[];
  returnType: PrimitiveType;
  wasmBytes: Uint8Array;
}

export interface CompiledSnUnit {
  irVersion: string;
  irHash: string;
  program: SnProgram;
  diagnostics: Diagnostic[];
  wasmFunctions: Map<string, CompiledWasmFunction>;
  rustFfiStub: string;
  tsBridgeCode: string;
}

export type SanitizerFn = (payload: unknown) => unknown;
export type NativeFfiHandler = (...args: number[]) => number | bigint;

export interface ControlAuditEntry {
  timestampMs: number;
  wireName: string;
  fromTopic: string;
  toTarget: string;
  status: "delivered" | "blocked_taint" | "blocked_invariant" | "error_routed";
  reason?: string;
}

export class SnError extends Error {
  constructor(
    message: string,
    public readonly code: string = "SN_ERROR",
    public readonly span?: SourceSpan,
  ) {
    super(message);
    this.name = "SnError";
  }
}
