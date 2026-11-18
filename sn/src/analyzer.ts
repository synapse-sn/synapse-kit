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
  const colonIdx = target.indexOf(":");
  if (colonIdx > 0) {
    return target.slice(0, colonIdx);
  }
  const dotIdx = target.indexOf(".");
  if (dotIdx > 0) {
    return target.slice(0, dotIdx);
  }
  return target;
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

    // 정책 및 CaMeL Taint Flow 검증
    const effectiveTaint: TaintLevel = wire.sanitizer ? "sanitized" : wire.sourceTaint;
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
        // 정책의 taint 규칙 확인
        for (const rule of policy.taintRules) {
          if (rule.sourceTaint === effectiveTaint && rule.deniedCapabilities.includes(targetCap)) {
            diagnostics.push({
              code: "TAINT_VIOLATION",
              severity: "error",
              message: SN_MESSAGES.TAINT_VIOLATION(wire.name, wire.toTarget, policy.name),
              span: wire.span,
            });
          }
        }
        // 전역 deny 목록 확인
        if (policy.deniedCapabilities.includes(targetCap)) {
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
      // 가드 정책 없이 비신뢰 입력이 민감 싱크로 직행하는 경우 차단
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
