/**
 * 시냅스 옵틱 유사 인터랙티브 노드 자동 진단기 (Locator Diagnostics)
 * - 액션/로케이터 매칭 실패 시 현재 A11y 스냅샷에서 가장 유사한 요소 후보를 추천
 * - 단축 별칭(@1)을 단발 CLI 호환 전체 세대 참조(@generation:1)로 자동 해석
 */
import { OPTIC_FLASH_DEFAULTS } from "./config";
import type { LocatorSuggestion, OpticSnapshot, ReferenceNode } from "./types";

export interface DiagnosticReport {
  readonly hint: string;
  readonly suggestions: readonly LocatorSuggestion[];
  readonly formattedMessage: string;
}

function normalizeText(input: string): string {
  return input
    .toLowerCase()
    .replace(/^[@#.\s]+/, "")
    .replace(/[-_]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

/**
 * 공간 복잡도 O(min(N, M)) 레벤슈타인 편집 거리 기반 정규화 유사도 (0..1)
 */
export function computeEditSimilarity(aRaw: string, bRaw: string): number {
  const a = normalizeText(aRaw);
  const b = normalizeText(bRaw);
  if (!a && !b) return 1;
  if (!a || !b) return 0;
  if (a === b) return 1;

  const n = a.length;
  const m = b.length;
  const prev = new Uint16Array(m + 1);
  const curr = new Uint16Array(m + 1);

  for (let j = 0; j <= m; j++) prev[j] = j;

  for (let i = 1; i <= n; i++) {
    curr[0] = i;
    const ca = a.charCodeAt(i - 1);
    for (let j = 1; j <= m; j++) {
      const cost = ca === b.charCodeAt(j - 1) ? 0 : 1;
      const del = prev[j] + 1;
      const ins = curr[j - 1] + 1;
      const sub = prev[j - 1] + cost;
      curr[j] = Math.min(del, ins, sub);
    }
    prev.set(curr);
  }

  const maxLen = Math.max(n, m);
  return Math.max(0, 1 - prev[m] / maxLen);
}

/**
 * 문자 바이그램 Jaccard 유사도 (0..1)
 */
function computeBigramSimilarity(aRaw: string, bRaw: string): number {
  const a = normalizeText(aRaw).replace(/\s+/g, "");
  const b = normalizeText(bRaw).replace(/\s+/g, "");
  if (!a || !b) return 0;
  if (a === b) return 1;
  if (a.length < 2 || b.length < 2) {
    return a === b ? 1 : 0;
  }

  const setA = new Set<string>();
  for (let i = 0; i <= a.length - 2; i++) setA.add(a.slice(i, i + 2));

  const setB = new Set<string>();
  for (let i = 0; i <= b.length - 2; i++) setB.add(b.slice(i, i + 2));

  let inter = 0;
  for (const bg of setA) {
    if (setB.has(bg)) inter++;
  }
  const union = setA.size + setB.size - inter;
  return union > 0 ? inter / union : 0;
}

function parseRoleAndQuery(rawTarget: string): { expectedRole?: string; queryText: string } {
  const trimmed = rawTarget.trim();
  const rolePrefixMatch = trimmed.match(/^([a-zA-Z가-힣]+)\s*:\s*(.+)$/);
  if (rolePrefixMatch && !trimmed.startsWith("@")) {
    return {
      expectedRole: rolePrefixMatch[1].toLowerCase(),
      queryText: rolePrefixMatch[2].trim(),
    };
  }
  return { queryText: trimmed };
}

export class LocatorDiagnostics {
  /**
   * 단축 별칭(@1) 또는 정확한 role:label 표현을 스냅샷의 정식 @세대:번호 참조로 해석
   */
  public static resolveTargetReference(
    target: string,
    snapshot?: OpticSnapshot | null
  ): string {
    if (!snapshot || snapshot.nodes.length === 0) {
      return target;
    }

    const trimmed = target.trim();

    // 1) @숫자 단축 별칭 -> @generation:index 정식 참조 변환
    const shortMatch = trimmed.match(/^@(\d+)$/);
    if (shortMatch) {
      const idx = Number(shortMatch[1]) - 1;
      if (idx >= 0 && idx < snapshot.nodes.length) {
        const ref = snapshot.nodes[idx].reference;
        return ref.startsWith("@") ? ref : `@${ref}`;
      }
      return target;
    }

    // 2) role:정확한라벨 형식 지원 (유일 매칭일 때만 안전하게 자동 치환)
    const { expectedRole, queryText } = parseRoleAndQuery(trimmed);
    if (expectedRole) {
      const normQ = queryText.toLowerCase();
      const matches = snapshot.nodes.filter(
        (n) => n.role.toLowerCase() === expectedRole && n.label.trim().toLowerCase() === normQ
      );
      if (matches.length === 1) {
        const ref = matches[0].reference;
        return ref.startsWith("@") ? ref : `@${ref}`;
      }
    }

    return target;
  }

  /**
   * 액션 실패 대상에 대해 원인 힌트와 유사 인터랙티브 요소 후보 목록 생성
   */
  public static diagnose(
    failedTarget: string,
    snapshot?: OpticSnapshot | null,
    maxSuggestions: number = OPTIC_FLASH_DEFAULTS.MAX_DIAGNOSTIC_SUGGESTIONS
  ): DiagnosticReport {
    const trimmed = (failedTarget || "").trim();
    if (!snapshot || snapshot.nodes.length === 0) {
      const hint = "현재 캐시된 페이지 스냅샷이 없습니다. 먼저 goto(url)로 최신 스냅샷을 수집하세요.";
      return { hint, suggestions: [], formattedMessage: hint };
    }

    let hint = `요청한 대상('${trimmed}')과 일치하는 인터랙티브 요소를 찾지 못했습니다.`;

    // 원인 분류 1: 단축 별칭 범위 초과
    const shortMatch = trimmed.match(/^@(\d+)$/);
    if (shortMatch) {
      const reqNum = Number(shortMatch[1]);
      if (reqNum < 1 || reqNum > snapshot.nodes.length) {
        hint = `단축 별칭 '${trimmed}'이 스냅샷 범위를 벗어났습니다 (유효 범위: @1 ~ @${snapshot.nodes.length}).`;
      }
    }

    // 원인 분류 2: 만료된 이전 세대(generation) 참조 사용
    const genMatch = trimmed.match(/^@?([a-fA-F0-9]{6,}):(\d+)$/);
    if (genMatch && snapshot.generation !== "unknown" && genMatch[1] !== snapshot.generation) {
      hint = `만료된 이전 스냅샷 세대('${genMatch[1]}') 참조입니다. 현재 활성 세대는 '${snapshot.generation}'입니다.`;
      const nodeIdx = Number(genMatch[2]) - 1;
      if (nodeIdx >= 0 && nodeIdx < snapshot.nodes.length) {
        const samePosNode = snapshot.nodes[nodeIdx];
        const suggestion: LocatorSuggestion = {
          reference: samePosNode.reference.startsWith("@")
            ? samePosNode.reference
            : `@${samePosNode.reference}`,
          shortAlias: samePosNode.shortAlias,
          role: samePosNode.role,
          label: samePosNode.label,
          similarity: 0.95,
        };
        const formattedMessage = `${hint} 동일 순번 후보: [${suggestion.shortAlias ?? suggestion.reference}] <${suggestion.role}> "${suggestion.label}"`;
        return { hint, suggestions: [suggestion], formattedMessage };
      }
    }

    // 유사 노드 점수 산출
    const { expectedRole, queryText } = parseRoleAndQuery(trimmed);
    const normQuery = normalizeText(queryText);

    const scored: LocatorSuggestion[] = [];
    for (const node of snapshot.nodes) {
      const score = LocatorDiagnostics.scoreCandidate(normQuery, expectedRole, node);
      if (score >= OPTIC_FLASH_DEFAULTS.MIN_DIAGNOSTIC_SIMILARITY) {
        scored.push({
          reference: node.reference.startsWith("@") ? node.reference : `@${node.reference}`,
          shortAlias: node.shortAlias,
          role: node.role,
          label: node.label,
          similarity: Number(score.toFixed(3)),
        });
      }
    }

    scored.sort((a, b) => b.similarity - a.similarity);
    const suggestions = scored.slice(0, Math.max(1, maxSuggestions));

    if (suggestions.length === 0) {
      return { hint, suggestions: [], formattedMessage: hint };
    }

    const suggestionLines = suggestions
      .map(
        (s) =>
          `[${s.shortAlias ?? s.reference}] <${s.role}> "${s.label}" (ref: ${s.reference}, 유사도 ${Math.round(s.similarity * 100)}%)`
      )
      .join(", ");

    return {
      hint,
      suggestions,
      formattedMessage: `${hint} 유사 요소 후보: ${suggestionLines}`,
    };
  }

  private static scoreCandidate(
    normQuery: string,
    expectedRole: string | undefined,
    node: ReferenceNode
  ): number {
    if (!normQuery) return 0;
    const normLabel = normalizeText(node.label);
    const normRole = normalizeText(node.role);

    const editSim = computeEditSimilarity(normQuery, normLabel);
    const bigramSim = computeBigramSimilarity(normQuery, normLabel);
    let score = editSim * 0.55 + bigramSim * 0.45;

    if (normLabel && (normLabel.includes(normQuery) || normQuery.includes(normLabel))) {
      score = Math.max(score, 0.72 + editSim * 0.2);
    }

    if (expectedRole) {
      if (normRole === expectedRole) {
        score += 0.15;
      } else {
        score *= 0.65;
      }
    } else if (normRole === normQuery) {
      score = Math.max(score, 0.55);
    }

    return Math.min(1, score);
  }
}
