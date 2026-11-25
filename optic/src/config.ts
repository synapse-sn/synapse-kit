/**
 * 시냅스 옵틱(Synapse Optic) 브라우저 엔진 설정 및 상수
 */
import { homedir } from "node:os";
import { join } from "node:path";

const USER_HOME = homedir();

export const OPTIC_DEFAULTS = {
  /** 기본 CDP 디버깅 호스트 엔드포인트 */
  DEFAULT_ENDPOINT: "http://127.0.0.1:9222",
  /** 기본 동작 제한 타임아웃 (ms) */
  DEFAULT_ACTION_TIMEOUT_MS: 30_000,
  /** 기본 네트워크 유휴 대기 시간 (ms) */
  DEFAULT_NETWORK_IDLE_MS: 3_000,
  /** 기본 스크롤 이동 픽셀 (px) */
  DEFAULT_SCROLL_AMOUNT: 600,
  /** 네이티브 프로메테우스 바이너리 탐색 후보 경로 */
  NATIVE_PROMETHEUS_CANDIDATES: [
    ...(process.env.PROMETHEUS_BIN ? [process.env.PROMETHEUS_BIN] : []),
    join(USER_HOME, ".gemini", "antigravity", "scratch", "project-prometheus", "target", "release", "prometheus-cli.exe"),
    join(USER_HOME, ".gemini", "antigravity", "scratch", "project-prometheus", "target", "debug", "prometheus-cli.exe"),
    "./target/release/prometheus-cli.exe",
    "prometheus-cli.exe",
    "prom.exe",
  ] as readonly string[],
  /** 상호작용 지원 표준 역할(Role) 집합 */
  INTERACTIVE_ROLES: new Set([
    "button",
    "link",
    "textbox",
    "searchbox",
    "checkbox",
    "combobox",
    "menuitem",
    "tab",
    "radio",
    "switch",
    "slider",
    "spinbutton",
    "listbox",
    "option",
    "heading",
  ]),
} as const;

/**
 * 시냅스 옵틱 하이브리드 라우터 및 액션 레코더 기본 상수
 */
export const OPTIC_HYBRID_DEFAULTS = {
  FAST_PATH_PATTERNS: ["arxiv.org", ".pdf", "github.com", "docs."] as const,
  TEXT_PAYLOAD_THRESHOLD: 500,
  DEFAULT_SESSION_NAME: "default_session",
} as const;

/**
 * 영속 REPL 세션, 유사 로케이터 진단, 적응형 정착(Adaptive Settle) 기본 상수
 */
export const OPTIC_FLASH_DEFAULTS = {
  /** 영속 REPL 초기 연결 제한 시간 (ms) */
  REPL_CONNECT_TIMEOUT_MS: 10_000,
  /** 영속 REPL 단일 명령 제한 시간 (ms) */
  REPL_COMMAND_TIMEOUT_MS: 15_000,
  /** 영속 세션 유휴 자동 정리 시간 (기본 5분) */
  SESSION_IDLE_TTL_MS: 300_000,
  /** 실패 시 최대 유사 요소 추천 개수 */
  MAX_DIAGNOSTIC_SUGGESTIONS: 3,
  /** 유사 요소 추천 최소 유사도 임계치 (0..1) */
  MIN_DIAGNOSTIC_SIMILARITY: 0.25,
  /** 적응형 정착(Adaptive Settle) 조용한 구간 기준 (ms) */
  ADAPTIVE_QUIET_WINDOW_MS: 80,
  /** 유휴 상태 조기 실패(Fail-Fast Idle) 최소 관찰 구간 (ms) */
  FAIL_FAST_IDLE_MS: 160,
  /** 적응형 정착 기본 최대 대기 상한 (ms) */
  ADAPTIVE_MAX_WAIT_MS: 2_000,
} as const;

export interface OpticConfig {
  /** CDP 엔드포인트 URL (기본: http://127.0.0.1:9222) */
  readonly endpoint?: string;
  /** 명시적 대상 브라우저 탭 타깃 ID */
  readonly targetId?: string;
  /** 네이티브 바이너리 실행 파일 경로 */
  readonly binaryPath?: string;
  /** 작업 제한 타임아웃 (ms) */
  readonly timeoutMs?: number;
  /** 헤드리스 모드 활성화 여부 */
  readonly headless?: boolean;
  /** 영속 세션 기본 활성화 여부 (프로세스 스폰 오버헤드 제거) */
  readonly persistentSession?: boolean;
  /** 영속 세션 백엔드 선택 ("direct_cdp": 상시 WebSocket Zero-Wait, "repl_pipe": Rust CLI REPL 파이프) */
  readonly sessionBackend?: "direct_cdp" | "repl_pipe";
  /** 액션 실패 시 유사 인터랙티브 노드 자동 진단 활성화 여부 (기본: true) */
  readonly enableDiagnostics?: boolean;
}
