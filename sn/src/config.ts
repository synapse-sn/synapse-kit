/**
 * sn 제어 언어 (Synapse Control Language) 전역 설정 및 상수 정의
 * - 모든 제한치, 기본값, 프로토콜 버전, 한국어 에러 메시지를 중앙 관리합니다.
 */

export const SN_DEFAULTS = {
  /** sn 제어 언어 IR 스키마 버전 */
  IR_VERSION: "0.1.0",
  /** 기본 실행 타임아웃 (밀리초) */
  DEFAULT_TIMEOUT_MS: 200,
  /** 기본 샌드박스 메모리 상한 (MB) */
  DEFAULT_MAX_MEMORY_MB: 32,
  /** 최대 AST 깊이 (스택 오버플로우 방어) */
  MAX_EXPR_DEPTH: 64,
  /** 최대 감사 로그 보관 개수 */
  MAX_AUDIT_ENTRIES: 4096,
  /** 순환 와이어 무한 재귀 방어 최대 홉 깊이 */
  MAX_WIRE_HOPS: 16,
  /** 멤버 접근 프로토타입 체인 차단 속성 */
  FORBIDDEN_MEMBER_PROPS: new Set<string>(["__proto__", "constructor", "prototype"]),
  /** 민감 싱크 기본 목록 (Taint 검증용) */
  SENSITIVE_SINKS: [
    "shell",
    "fs_write",
    "native_ffi",
    "net_egress",
    "process_spawn",
  ] as const,
} as const;

export const SN_MESSAGES = {
  UNEXPECTED_TOKEN: (expected: string, actual: string, line: number, col: number) =>
    `[sn] 구문 오류 (${line}:${col}): '${expected}'이(가) 필요하지만 '${actual}'이(가) 입력되었습니다.`,
  UNTERMINATED_STRING: (line: number, col: number) =>
    `[sn] 어휘 오류 (${line}:${col}): 닫히지 않은 문자열 리터럴입니다.`,
  UNEXPECTED_CHAR: (ch: string, line: number, col: number) =>
    `[sn] 어휘 오류 (${line}:${col}): 지원하지 않는 문자 '${ch}'입니다.`,
  EXPR_TOO_DEEP: (maxDepth: number) =>
    `[sn] 표현식 중첩 깊이 한도(${maxDepth})를 초과했습니다.`,
  TAINT_VIOLATION: (wireName: string, target: string, policyName: string) =>
    `[sn] 보안 오염 위반 (Taint Violation): 와이어 '${wireName}'의 비신뢰(untrusted) 데이터가 sanitize 없이 보호 대상 '${target}'(정책 '${policyName}')로 전달될 수 없습니다.`,
  UNKNOWN_POLICY: (wireName: string, policyName: string) =>
    `[sn] 참조 오류: 와이어 '${wireName}'가 존재하지 않는 정책 '${policyName}'을(를) 참조합니다.`,
  UNKNOWN_INVARIANT: (wireName: string, invName: string) =>
    `[sn] 참조 오류: 와이어 '${wireName}'가 존재하지 않는 불변식 '${invName}'을(를) 참조합니다.`,
  DUPLICATE_DECL: (kind: string, name: string) =>
    `[sn] 중복 선언 오류: ${kind} '${name}'이(가) 이미 정의되어 있습니다.`,
  UNSUPPORTED_WASM_EXPR: (detail: string) =>
    `[sn] WASM 컴파일 오류: ${detail}`,
} as const;
