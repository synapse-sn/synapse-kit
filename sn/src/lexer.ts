/**
 * 시냅스 제어 언어 어휘 분석기 (Deterministic Lexer)
 * - 라인/컬럼 위치를 정확히 추적해 진단 메시지에 제공합니다.
 */
import { SN_MESSAGES } from "./config";
import { SnError, type Token } from "./types";

const KEYWORDS = new Set([
  "policy",
  "native",
  "fn",
  "invariant",
  "wire",
  "via",
  "wasm",
  "ffi",
  "bus",
  "taint",
  "untrusted",
  "trusted",
  "sanitized",
  "allow",
  "deny",
  "quota",
  "cause",
  "require",
  "effect",
  "severity",
  "critical",
  "warn",
  "from",
  "as",
  "guard",
  "sanitize",
  "check",
  "to",
  "on_error",
  "return",
  "true",
  "false",
  "i32",
  "i64",
  "bool",
]);

const TWO_CHAR_OPS = new Set(["->", "==", "!=", ">=", "<=", "&&", "||"]);
const ONE_CHAR_OPS = new Set(["+", "-", "*", "/", "%", ">", "<", "!", "="]);
const PUNCT_CHARS = new Set(["{", "}", "(", ")", ":", ";", ",", "."]);

export function tokenize(source: string): Token[] {
  const tokens: Token[] = [];
  let pos = 0;
  let line = 1;
  let column = 1;
  const len = source.length;

  const advance = (count = 1): void => {
    for (let i = 0; i < count; i++) {
      if (source[pos] === "\n") {
        line++;
        column = 1;
      } else {
        column++;
      }
      pos++;
    }
  };

  while (pos < len) {
    const ch = source[pos];

    // 공백 건너뛰기
    if (ch === " " || ch === "\t" || ch === "\r" || ch === "\n") {
      advance();
      continue;
    }

    // 한 줄 주석 (// 또는 #)
    if ((ch === "/" && source[pos + 1] === "/") || ch === "#") {
      while (pos < len && source[pos] !== "\n") {
        advance();
      }
      continue;
    }

    // 블록 주석 (/* ... */)
    if (ch === "/" && source[pos + 1] === "*") {
      advance(2);
      while (pos < len && !(source[pos] === "*" && source[pos + 1] === "/")) {
        advance();
      }
      if (pos < len) {
        advance(2);
      }
      continue;
    }

    const startLine = line;
    const startCol = column;

    // 문자열 리터럴
    if (ch === '"') {
      advance();
      let str = "";
      while (pos < len && source[pos] !== '"') {
        if (source[pos] === "\\") {
          advance();
          if (pos >= len) {
            throw new SnError(
              SN_MESSAGES.UNTERMINATED_STRING(startLine, startCol),
              "LEXER_UNTERMINATED_STRING",
              { line: startLine, column: startCol },
            );
          }
          const esc = source[pos];
          if (esc === "n") str += "\n";
          else if (esc === "t") str += "\t";
          else if (esc === '"') str += '"';
          else if (esc === "\\") str += "\\";
          else str += esc;
          advance();
        } else {
          str += source[pos];
          advance();
        }
      }
      if (pos >= len) {
        throw new SnError(
          SN_MESSAGES.UNTERMINATED_STRING(startLine, startCol),
          "LEXER_UNTERMINATED_STRING",
          { line: startLine, column: startCol },
        );
      }
      advance();
      tokens.push({ kind: "STRING", value: str, line: startLine, column: startCol });
      continue;
    }

    // 숫자 리터럴
    if (ch >= "0" && ch <= "9") {
      let numStr = "";
      let hasDot = false;
      while (pos < len) {
        const cur = source[pos];
        if (cur >= "0" && cur <= "9") {
          numStr += cur;
          advance();
        } else if (cur === "." && !hasDot && source[pos + 1] >= "0" && source[pos + 1] <= "9") {
          hasDot = true;
          numStr += cur;
          advance();
        } else {
          break;
        }
      }
      tokens.push({ kind: "NUMBER", value: numStr, line: startLine, column: startCol });
      continue;
    }

    // 식별자 및 키워드
    if ((ch >= "a" && ch <= "z") || (ch >= "A" && ch <= "Z") || ch === "_") {
      let ident = "";
      while (pos < len) {
        const cur = source[pos];
        if (
          (cur >= "a" && cur <= "z") ||
          (cur >= "A" && cur <= "Z") ||
          (cur >= "0" && cur <= "9") ||
          cur === "_"
        ) {
          ident += cur;
          advance();
        } else {
          break;
        }
      }
      const kind = KEYWORDS.has(ident) ? "KEYWORD" : "IDENT";
      tokens.push({ kind, value: ident, line: startLine, column: startCol });
      continue;
    }

    // 2글자 연산자
    const two = source.slice(pos, pos + 2);
    if (TWO_CHAR_OPS.has(two)) {
      advance(2);
      tokens.push({ kind: "OP", value: two, line: startLine, column: startCol });
      continue;
    }

    // 1글자 연산자
    if (ONE_CHAR_OPS.has(ch)) {
      advance();
      tokens.push({ kind: "OP", value: ch, line: startLine, column: startCol });
      continue;
    }

    // 구두점
    if (PUNCT_CHARS.has(ch)) {
      advance();
      tokens.push({ kind: "PUNCT", value: ch, line: startLine, column: startCol });
      continue;
    }

    throw new SnError(
      SN_MESSAGES.UNEXPECTED_CHAR(ch, startLine, startCol),
      "LEXER_UNEXPECTED_CHAR",
      { line: startLine, column: startCol },
    );
  }

  tokens.push({ kind: "EOF", value: "", line, column });
  return tokens;
}
