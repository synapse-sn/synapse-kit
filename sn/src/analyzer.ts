/**
 * 시냅스 제어 언어 정적 의미 분석기 & CaMeL 기반 Taint/Capability 검증기
 * - 중복 선언, 미정의 참조, 비신뢰(untrusted) 데이터의 민감 싱크 무검증 유입을 컴파일 타임에 차단합니다.
 */
import { SN_DEFAULTS, SN_MESSAGES } from "./config";
import type {
  Diagnostic,
  InvariantDecl,
  NativeFnDecl,
  PolicyDecl,
  SnProgram,
  TaintLevel,
  WireDecl,
} from "./types";

function extractCapabilityFromTarget(target: string): string {
  // 예: "shell:exec" -> "shell", "fs_write:/tmp/a" -> "fs_write", "native_ffi:hash" -> "native_ffi"
  const normalized = target.trim().toLowerCase();
  const colonIdx = normalized.indexOf(":");
  if (colonIdx > 0) {
    return normalized.slice(0, colonIdx).trim();
  }
  const dotIdx = normalized.indexOf(".");
  if (dotIdx > 0) {
    return normalized.slice(0, dotIdx).trim();
  }
  return normalized;
}

export function analyzeProgram(program: SnProgram): Diagnostic[] {
  const diagnostics: Diagnostic[] = [];

  const policies = new Map<string, PolicyDecl>();
  const nativeFns = new Map<string, NativeFnDecl>();
  const invariants = new Map<string, InvariantDecl>();
  const wires = new Map<string, WireDecl>();

  for (const p of program.policies) {
    if (policies.has(p.name)) {
      diagnostics.push({
        code: "DUPLICATE_POLICY",
        severity: "error",
        message: SN_MESSAGES.DUPLICATE_DECL("정책", p.name),
        span: p.span,
      });
    } else {
      policies.set(p.name, p);
    }
  }

  for (const fn of program.nativeFns) {
    if (nativeFns.has(fn.name)) {
      diagnostics.push({
        code: "DUPLICATE_NATIVE_FN",
        severity: "error",
        message: SN_MESSAGES.DUPLICATE_DECL("네이티브 함수", fn.name),
        span: fn.span,
      });
    } else {
      nativeFns.set(fn.name, fn);
    }
  }

  for (const inv of program.invariants) {
    if (invariants.has(inv.name)) {
      diagnostics.push({
        code: "DUPLICATE_INVARIANT",
        severity: "error",
        message: SN_MESSAGES.DUPLICATE_DECL("불변식", inv.name),
        span: inv.span,
      });
    } else {
      invariants.set(inv.name, inv);
    }
  }

  // 1차 패스: 중복 선언 및 미정의 참조 검증 + 와이어 목록 수집
  const validWires: WireDecl[] = [];
  const topicTaint = new Map<string, TaintLevel>();

  for (const wire of program.wires) {
    if (wires.has(wire.name)) {
      diagnostics.push({
        code: "DUPLICATE_WIRE",
        severity: "error",
        message: SN_MESSAGES.DUPLICATE_DECL("와이어", wire.name),
        span: wire.span,
      });
      continue;
    }
    wires.set(wire.name, wire);
    validWires.push(wire);

    // 불변식 참조 검증
    for (const checkName of wire.checks) {
      if (!invariants.has(checkName)) {
        diagnostics.push({
          code: "UNKNOWN_INVARIANT",
          severity: "error",
          message: SN_MESSAGES.UNKNOWN_INVARIANT(wire.name, checkName),
          span: wire.span,
        });
      }
    }

    if (wire.sourceTaint === "untrusted") {
      topicTaint.set(wire.fromTopic, "untrusted");
    }
  }

  // 2차 패스: 다중 홉 CaMeL Taint 고정점 전파 (sanitize 없는 와이어를 거치면 타깃 토픽도 untrusted 전파)
  const maxIterations = validWires.length + 1;
  for (let iter = 0; iter < maxIterations; iter++) {
    let changed = false;
    for (const wire of validWires) {
      const inheritedTaint: TaintLevel =
        wire.sourceTaint === "untrusted" || topicTaint.get(wire.fromTopic) === "untrusted"
          ? "untrusted"
          : wire.sourceTaint;
      const propagated: TaintLevel = wire.sanitizer ? "sanitized" : inheritedTaint;
      if (propagated === "untrusted" && topicTaint.get(wire.toTarget) !== "untrusted") {
        topicTaint.set(wire.toTarget, "untrusted");
        changed = true;
      }
    }
    if (!changed) break;
  }

  // 3차 패스: 전파된 유효 Taint 기준 정책 및 민감 싱크 위반 검사
  for (const wire of validWires) {
    const inheritedTaint: TaintLevel =
      wire.sourceTaint === "untrusted" || topicTaint.get(wire.fromTopic) === "untrusted"
        ? "untrusted"
        : wire.sourceTaint;
    const effectiveTaint: TaintLevel = wire.sanitizer ? "sanitized" : inheritedTaint;
    const targetCap = extractCapabilityFromTarget(wire.toTarget);

    if (wire.guardPolicy) {
      const policy = policies.get(wire.guardPolicy);
      if (!policy) {
        diagnostics.push({
          code: "UNKNOWN_POLICY",
          severity: "error",
          message: SN_MESSAGES.UNKNOWN_POLICY(wire.name, wire.guardPolicy),
          span: wire.span,
        });
      } else {
        for (const rule of policy.taintRules) {
          const deniedSet = rule.deniedCapabilities.map((c) => c.trim().toLowerCase());
          if (rule.sourceTaint === effectiveTaint && deniedSet.includes(targetCap)) {
            diagnostics.push({
              code: "TAINT_VIOLATION",
              severity: "error",
              message: SN_MESSAGES.TAINT_VIOLATION(wire.name, wire.toTarget, policy.name),
              span: wire.span,
            });
          }
        }
        const globalDenied = policy.deniedCapabilities.map((c) => c.trim().toLowerCase());
        if (globalDenied.includes(targetCap)) {
          diagnostics.push({
            code: "CAPABILITY_DENIED",
            severity: "error",
            message: SN_MESSAGES.TAINT_VIOLATION(wire.name, wire.toTarget, policy.name),
            span: wire.span,
          });
        }
      }
    } else if (
      effectiveTaint === "untrusted" &&
      (SN_DEFAULTS.SENSITIVE_SINKS as readonly string[]).includes(targetCap)
    ) {
      // 가드 정책 없이 비신뢰 입력(직접 또는 다중 홉 전파)이 민감 싱크로 흐르는 경우 차단
      diagnostics.push({
        code: "TAINT_VIOLATION",
        severity: "error",
        message: SN_MESSAGES.TAINT_VIOLATION(wire.name, wire.toTarget, "<none>"),
        span: wire.span,
      });
    }
  }

  return diagnostics;
}
