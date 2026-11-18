/**
 * 시냅스 제어 언어 재귀 하강 파서 (Recursive Descent Parser)
 * - policy / native fn / invariant / wire 선언 및 표현식 AST를 생성합니다.
 */
import { SN_DEFAULTS, SN_MESSAGES } from "./config";
import { tokenize } from "./lexer";
import {
  SnError,
  type Expr,
  type InvariantDecl,
  type NativeFnDecl,
  type NativeParam,
  type NativeTransport,
  type PolicyDecl,
  type PrimitiveType,
  type SeverityLevel,
  type SourceSpan,
  type SnProgram,
  type TaintLevel,
  type TaintRule,
  type Token,
  type WireDecl,
} from "./types";

class Parser {
  private index = 0;

  constructor(private readonly tokens: Token[]) {}

  private peek(offset = 0): Token {
    return this.tokens[this.index + offset] ?? this.tokens[this.tokens.length - 1];
  }

  private advance(): Token {
    const tok = this.peek();
    if (tok.kind !== "EOF") {
      this.index++;
    }
    return tok;
  }

  private check(value: string): boolean {
    return this.peek().value === value && this.peek().kind !== "EOF";
  }

  private expect(value: string): Token {
    const tok = this.peek();
    if (tok.value !== value) {
      throw new SnError(
        SN_MESSAGES.UNEXPECTED_TOKEN(value, tok.value || "EOF", tok.line, tok.column),
        "PARSE_UNEXPECTED_TOKEN",
        { line: tok.line, column: tok.column },
      );
    }
    return this.advance();
  }

  private expectIdent(label = "식별자"): Token {
    const tok = this.peek();
    if (tok.kind !== "IDENT" && tok.kind !== "KEYWORD") {
      throw new SnError(
        SN_MESSAGES.UNEXPECTED_TOKEN(label, tok.value || "EOF", tok.line, tok.column),
        "PARSE_EXPECTED_IDENT",
        { line: tok.line, column: tok.column },
      );
    }
    return this.advance();
  }

  private expectString(label = "문자열"): string {
    const tok = this.peek();
    if (tok.kind !== "STRING") {
      throw new SnError(
        SN_MESSAGES.UNEXPECTED_TOKEN(label, tok.value || "EOF", tok.line, tok.column),
        "PARSE_EXPECTED_STRING",
        { line: tok.line, column: tok.column },
      );
    }
    return this.advance().value;
  }

  private expectNumber(label = "숫자"): number {
    const tok = this.peek();
    if (tok.kind !== "NUMBER") {
      throw new SnError(
        SN_MESSAGES.UNEXPECTED_TOKEN(label, tok.value || "EOF", tok.line, tok.column),
        "PARSE_EXPECTED_NUMBER",
        { line: tok.line, column: tok.column },
      );
    }
    return Number(this.advance().value);
  }

  private spanOf(tok: Token): SourceSpan {
    return { line: tok.line, column: tok.column };
  }

  public parseProgram(): SnProgram {
    const policies: PolicyDecl[] = [];
    const nativeFns: NativeFnDecl[] = [];
    const invariants: InvariantDecl[] = [];
    const wires: WireDecl[] = [];

    while (this.peek().kind !== "EOF") {
      const tok = this.peek();
      if (tok.value === "policy") {
        policies.push(this.parsePolicy());
      } else if (tok.value === "native") {
        nativeFns.push(this.parseNativeFn());
      } else if (tok.value === "invariant") {
        invariants.push(this.parseInvariant());
      } else if (tok.value === "wire") {
        wires.push(this.parseWire());
      } else {
        throw new SnError(
          SN_MESSAGES.UNEXPECTED_TOKEN(
            "policy | native | invariant | wire",
            tok.value,
            tok.line,
            tok.column,
          ),
          "PARSE_UNKNOWN_TOP_LEVEL",
          this.spanOf(tok),
        );
      }
    }

    return {
      kind: "SnProgram",
      policies,
      nativeFns,
      invariants,
      wires,
    };
  }

  private parseIdentList(): string[] {
    const list: string[] = [];
    list.push(this.expectIdent().value);
    while (this.check(",")) {
      this.advance();
      list.push(this.expectIdent().value);
    }
    return list;
  }

  private parsePolicy(): PolicyDecl {
    const start = this.expect("policy");
    const name = this.expectIdent("정책 이름").value;
    this.expect("{");

    const taintRules: TaintRule[] = [];
    const allowedCapabilities: string[] = [];
    const deniedCapabilities: string[] = [];
    const quotas: Record<string, number> = {};

    while (!this.check("}") && this.peek().kind !== "EOF") {
      const kw = this.peek();
      if (kw.value === "taint") {
        this.advance();
        const taintTok = this.expectIdent("오염 등급");
        const sourceTaint = taintTok.value as TaintLevel;
        if (sourceTaint !== "untrusted" && sourceTaint !== "trusted" && sourceTaint !== "sanitized") {
          throw new SnError(
            SN_MESSAGES.UNEXPECTED_TOKEN("untrusted|trusted|sanitized", taintTok.value, taintTok.line, taintTok.column),
            "PARSE_INVALID_TAINT",
            this.spanOf(taintTok),
          );
        }
        this.expect("->");
        this.expect("deny");
        const denied = this.parseIdentList();
        this.expect(";");
        taintRules.push({ sourceTaint, deniedCapabilities: denied });
      } else if (kw.value === "allow") {
        this.advance();
        allowedCapabilities.push(...this.parseIdentList());
        this.expect(";");
      } else if (kw.value === "deny") {
        this.advance();
        deniedCapabilities.push(...this.parseIdentList());
        this.expect(";");
      } else if (kw.value === "quota") {
        this.advance();
        const qKey = this.expectIdent("쿼터 키").value;
        this.expect("=");
        const qVal = this.expectNumber("쿼터 값");
        this.expect(";");
        quotas[qKey] = qVal;
      } else {
        throw new SnError(
          SN_MESSAGES.UNEXPECTED_TOKEN("taint | allow | deny | quota", kw.value, kw.line, kw.column),
          "PARSE_INVALID_POLICY_STMT",
          this.spanOf(kw),
        );
      }
    }

    this.expect("}");
    return {
      kind: "PolicyDecl",
      name,
      taintRules,
      allowedCapabilities,
      deniedCapabilities,
      quotas,
      span: this.spanOf(start),
    };
  }

  private parsePrimitiveType(): PrimitiveType {
    const tok = this.expectIdent("원시 타입");
    if (tok.value !== "i32" && tok.value !== "i64" && tok.value !== "bool") {
      throw new SnError(
        SN_MESSAGES.UNEXPECTED_TOKEN("i32 | i64 | bool", tok.value, tok.line, tok.column),
        "PARSE_INVALID_TYPE",
        this.spanOf(tok),
      );
    }
    return tok.value;
  }

  private parseNativeFn(): NativeFnDecl {
    const start = this.expect("native");
    this.expect("fn");
    const name = this.expectIdent("함수 이름").value;
    this.expect("(");

    const params: NativeParam[] = [];
    if (!this.check(")")) {
      do {
        if (this.check(",")) this.advance();
        const pName = this.expectIdent("매개변수 이름").value;
        this.expect(":");
        const pType = this.parsePrimitiveType();
        params.push({ name: pName, type: pType });
      } while (this.check(","));
    }
    this.expect(")");
    this.expect("->");
    const returnType = this.parsePrimitiveType();
    this.expect("via");

    const trTok = this.expectIdent("전송 계층");
    const transport = trTok.value as NativeTransport;
    if (transport !== "wasm" && transport !== "ffi" && transport !== "bus") {
      throw new SnError(
        SN_MESSAGES.UNEXPECTED_TOKEN("wasm | ffi | bus", trTok.value, trTok.line, trTok.column),
        "PARSE_INVALID_TRANSPORT",
        this.spanOf(trTok),
      );
    }

    let targetRef: string | undefined;
    let bodyExpr: Expr | undefined;

    if (transport === "wasm" && this.check("{")) {
      this.expect("{");
      this.expect("return");
      bodyExpr = this.parseExpr(0);
      this.expect(";");
      this.expect("}");
    } else {
      targetRef = this.expectString("네이티브 심볼/라이브러리 경로");
      this.expect(";");
    }

    return {
      kind: "NativeFnDecl",
      name,
      params,
      returnType,
      transport,
      targetRef,
      bodyExpr,
      span: this.spanOf(start),
    };
  }

  private parseInvariant(): InvariantDecl {
    const start = this.expect("invariant");
    const name = this.expectIdent("불변식 이름").value;
    this.expect("{");

    let cause = "";
    let requireExpr: Expr | null = null;
    let effect = "";
    let severity: SeverityLevel = "critical";

    while (!this.check("}") && this.peek().kind !== "EOF") {
      const kw = this.peek();
      if (kw.value === "cause") {
        this.advance();
        cause = this.expectString("원인 설명");
        this.expect(";");
      } else if (kw.value === "require") {
        this.advance();
        requireExpr = this.parseExpr(0);
        this.expect(";");
      } else if (kw.value === "effect") {
        this.advance();
        effect = this.expectString("효과 설명");
        this.expect(";");
      } else if (kw.value === "severity") {
        this.advance();
        const sev = this.expectIdent("심각도");
        if (sev.value !== "critical" && sev.value !== "warn") {
          throw new SnError(
            SN_MESSAGES.UNEXPECTED_TOKEN("critical | warn", sev.value, sev.line, sev.column),
            "PARSE_INVALID_SEVERITY",
            this.spanOf(sev),
          );
        }
        severity = sev.value;
        this.expect(";");
      } else {
        throw new SnError(
          SN_MESSAGES.UNEXPECTED_TOKEN("cause | require | effect | severity", kw.value, kw.line, kw.column),
          "PARSE_INVALID_INVARIANT_STMT",
          this.spanOf(kw),
        );
      }
    }

    this.expect("}");
    if (!requireExpr) {
      throw new SnError(
        `[sn] 불변식 '${name}'에는 반드시 'require <조건식>;'이 포함되어야 합니다.`,
        "PARSE_MISSING_REQUIRE",
        this.spanOf(start),
      );
    }

    return {
      kind: "InvariantDecl",
      name,
      cause,
      requireExpr,
      effect,
      severity,
      span: this.spanOf(start),
    };
  }

  private parseWire(): WireDecl {
    const start = this.expect("wire");
    const name = this.expectIdent("와이어 이름").value;
    this.expect("{");

    let fromTopic = "";
    let sourceTaint: TaintLevel = "trusted";
    let guardPolicy: string | undefined;
    let sanitizer: string | undefined;
    const checks: string[] = [];
    let toTarget = "";
    let onErrorTopic: string | undefined;

    while (!this.check("}") && this.peek().kind !== "EOF") {
      const kw = this.peek();
      if (kw.value === "from") {
        this.advance();
        fromTopic = this.expectString("소스 토픽");
        if (this.check("as")) {
          this.advance();
          const tTok = this.expectIdent("오염 등급");
          if (tTok.value !== "trusted" && tTok.value !== "untrusted" && tTok.value !== "sanitized") {
            throw new SnError(
              SN_MESSAGES.UNEXPECTED_TOKEN("trusted | untrusted | sanitized", tTok.value, tTok.line, tTok.column),
              "PARSE_INVALID_TAINT",
              this.spanOf(tTok),
            );
          }
          sourceTaint = tTok.value;
        }
        this.expect(";");
      } else if (kw.value === "guard") {
        this.advance();
        guardPolicy = this.expectIdent("정책 이름").value;
        this.expect(";");
      } else if (kw.value === "sanitize") {
        this.advance();
        sanitizer = this.expectIdent("새니타이저 이름").value;
        this.expect(";");
      } else if (kw.value === "check") {
        this.advance();
        checks.push(...this.parseIdentList());
        this.expect(";");
      } else if (kw.value === "to") {
        this.advance();
        toTarget = this.expectString("타겟 토픽/싱크");
        this.expect(";");
      } else if (kw.value === "on_error") {
        this.advance();
        onErrorTopic = this.expectString("폴백 토픽");
        this.expect(";");
      } else {
        throw new SnError(
          SN_MESSAGES.UNEXPECTED_TOKEN(
            "from | guard | sanitize | check | to | on_error",
            kw.value,
            kw.line,
            kw.column,
          ),
          "PARSE_INVALID_WIRE_STMT",
          this.spanOf(kw),
        );
      }
    }

    this.expect("}");
    if (!fromTopic || !toTarget) {
      throw new SnError(
        `[sn] 와이어 '${name}'에는 'from'과 'to'가 필수입니다.`,
        "PARSE_INCOMPLETE_WIRE",
        this.spanOf(start),
      );
    }

    return {
      kind: "WireDecl",
      name,
      fromTopic,
      sourceTaint,
      guardPolicy,
      sanitizer,
      checks,
      toTarget,
      onErrorTopic,
      span: this.spanOf(start),
    };
  }

  // --- 표현식 파서 (우선순위 기반) ---
  private parseExpr(depth: number): Expr {
    if (depth > SN_DEFAULTS.MAX_EXPR_DEPTH) {
      throw new SnError(
        SN_MESSAGES.EXPR_TOO_DEEP(SN_DEFAULTS.MAX_EXPR_DEPTH),
        "PARSE_EXPR_TOO_DEEP",
        this.spanOf(this.peek()),
      );
    }
    return this.parseOr(depth + 1);
  }

  private parseOr(depth: number): Expr {
    let left = this.parseAnd(depth);
    while (this.check("||")) {
      const op = this.advance();
      const right = this.parseAnd(depth);
      left = { kind: "Binary", op: "||", left, right, span: this.spanOf(op) };
    }
    return left;
  }

  private parseAnd(depth: number): Expr {
    let left = this.parseEquality(depth);
    while (this.check("&&")) {
      const op = this.advance();
      const right = this.parseEquality(depth);
      left = { kind: "Binary", op: "&&", left, right, span: this.spanOf(op) };
    }
    return left;
  }

  private parseEquality(depth: number): Expr {
    let left = this.parseComparison(depth);
    while (this.check("==") || this.check("!=")) {
      const op = this.advance();
      const right = this.parseComparison(depth);
      left = { kind: "Binary", op: op.value as "==" | "!=", left, right, span: this.spanOf(op) };
    }
    return left;
  }

  private parseComparison(depth: number): Expr {
    let left = this.parseTerm(depth);
    while (this.check("<") || this.check("<=") || this.check(">") || this.check(">=")) {
      const op = this.advance();
      const right = this.parseTerm(depth);
      left = {
        kind: "Binary",
        op: op.value as "<" | "<=" | ">" | ">=",
        left,
        right,
        span: this.spanOf(op),
      };
    }
    return left;
  }

  private parseTerm(depth: number): Expr {
    let left = this.parseFactor(depth);
    while (this.check("+") || this.check("-")) {
      const op = this.advance();
      const right = this.parseFactor(depth);
      left = { kind: "Binary", op: op.value as "+" | "-", left, right, span: this.spanOf(op) };
    }
    return left;
  }

  private parseFactor(depth: number): Expr {
    let left = this.parseUnary(depth);
    while (this.check("*") || this.check("/") || this.check("%")) {
      const op = this.advance();
      const right = this.parseUnary(depth);
      left = {
        kind: "Binary",
        op: op.value as "*" | "/" | "%",
        left,
        right,
        span: this.spanOf(op),
      };
    }
    return left;
  }

  private parseUnary(depth: number): Expr {
    if (this.check("!") || this.check("-")) {
      const op = this.advance();
      const operand = this.parseUnary(depth);
      return { kind: "Unary", op: op.value as "!" | "-", operand, span: this.spanOf(op) };
    }
    return this.parsePostfix(depth);
  }

  private parsePostfix(depth: number): Expr {
    let expr = this.parsePrimary(depth);
    while (true) {
      if (this.check(".")) {
        const dot = this.advance();
        const prop = this.expectIdent("속성 이름").value;
        expr = { kind: "MemberAccess", object: expr, property: prop, span: this.spanOf(dot) };
      } else if (this.check("(") && expr.kind === "Identifier") {
        this.advance();
        const args: Expr[] = [];
        if (!this.check(")")) {
          do {
            if (this.check(",")) this.advance();
            args.push(this.parseExpr(depth + 1));
          } while (this.check(","));
        }
        this.expect(")");
        expr = { kind: "Call", callee: expr.name, args, span: expr.span };
      } else {
        break;
      }
    }
    return expr;
  }

  private parsePrimary(depth: number): Expr {
    const tok = this.peek();
    if (tok.kind === "NUMBER") {
      this.advance();
      return {
        kind: "NumberLiteral",
        value: Number(tok.value),
        isFloat: tok.value.includes("."),
        span: this.spanOf(tok),
      };
    }
    if (tok.value === "true" || tok.value === "false") {
      this.advance();
      return {
        kind: "BoolLiteral",
        value: tok.value === "true",
        span: this.spanOf(tok),
      };
    }
    if (tok.kind === "STRING") {
      this.advance();
      return { kind: "StringLiteral", value: tok.value, span: this.spanOf(tok) };
    }
    if (tok.kind === "IDENT") {
      this.advance();
      return { kind: "Identifier", name: tok.value, span: this.spanOf(tok) };
    }
    if (this.check("(")) {
      this.advance();
      const inner = this.parseExpr(depth + 1);
      this.expect(")");
      return inner;
    }

    throw new SnError(
      SN_MESSAGES.UNEXPECTED_TOKEN("표현식", tok.value || "EOF", tok.line, tok.column),
      "PARSE_INVALID_EXPR",
      this.spanOf(tok),
    );
  }
}

export function parseSn(source: string): SnProgram {
  const tokens = tokenize(source);
  const parser = new Parser(tokens);
  return parser.parseProgram();
}
