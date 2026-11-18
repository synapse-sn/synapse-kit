#!/usr/bin/env bun
/**
 * 시냅스 제어 언어 CLI (check / compile)
 */
import { readFileSync } from "node:fs";
import { compileSn } from "./src/codegen";
import { SnError } from "./src/types";

function printOut(text: string): void {
  process.stdout.write(`${text}\n`);
}

function printErr(text: string): void {
  process.stderr.write(`${text}\n`);
}

function main(): void {
  const args = process.argv.slice(2);
  const cmd = args[0];
  const filePath = args[1];

  if (!cmd || !filePath || (cmd !== "check" && cmd !== "compile")) {
    printErr("사용법: bun run sn/cli.ts <check|compile> <파일.syn> [--target ir|rust|ts|wasm]");
    process.exit(1);
  }

  let source: string;
  try {
    source = readFileSync(filePath, "utf8");
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    printErr(`[sn] 파일을 읽을 수 없습니다 (${filePath}): ${msg}`);
    process.exit(1);
  }

  try {
    const unit = compileSn(source, { strict: true });
    if (cmd === "check") {
      printOut(
        JSON.stringify(
          {
            ok: true,
            irVersion: unit.irVersion,
            irHash: unit.irHash,
            policies: unit.program.policies.length,
            nativeFns: unit.program.nativeFns.length,
            invariants: unit.program.invariants.length,
            wires: unit.program.wires.length,
          },
          null,
          2,
        ),
      );
      return;
    }

    const targetIdx = args.indexOf("--target");
    const target = targetIdx !== -1 ? args[targetIdx + 1] : "ir";

    if (target === "rust") {
      printOut(unit.rustFfiStub);
    } else if (target === "ts") {
      printOut(unit.tsBridgeCode);
    } else if (target === "wasm") {
      const summary = Array.from(unit.wasmFunctions.entries()).map(([name, fn]) => ({
        name,
        bytes: fn.wasmBytes.length,
        hex: Array.from(fn.wasmBytes)
          .map((b) => b.toString(16).padStart(2, "0"))
          .join(" "),
      }));
      printOut(JSON.stringify(summary, null, 2));
    } else {
      printOut(
        JSON.stringify(
          {
            irVersion: unit.irVersion,
            irHash: unit.irHash,
            program: unit.program,
          },
          null,
          2,
        ),
      );
    }
  } catch (err) {
    if (err instanceof SnError) {
      printErr(`${err.message} (${err.code})`);
    } else {
      printErr(err instanceof Error ? err.message : String(err));
    }
    process.exit(1);
  }
}

main();
