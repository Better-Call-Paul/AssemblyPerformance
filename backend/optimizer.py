#!/usr/bin/env python3

import os
import sys
import json
import subprocess
import tempfile
import time
import hashlib
import statistics
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse

try:
    from openai import OpenAI
except ImportError:
    print("Please install openai: pip install openai")
    sys.exit(1)

class COptimizer:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found")
        self.client = OpenAI(api_key=self.api_key)
        self.work_dir = Path(tempfile.mkdtemp(prefix="c_opt_"))
        self.iterations = []
        
    def compile_and_analyze(self, c_code: str, flags: str = "-O3") -> Dict:
        src_file = self.work_dir / "func.c"
        src_file.write_text(c_code)
        
        obj_file = self.work_dir / "func.o"
        asm_file = self.work_dir / "func.s"
        ir_file = self.work_dir / "func.ll"
        
        result = {
            "source": c_code,
            "flags": flags,
            "success": False
        }
        
        try:
            subprocess.run(
                f"clang {flags} -c -o {obj_file} {src_file}",
                shell=True, check=True, capture_output=True, text=True
            )
            
            subprocess.run(
                f"clang {flags} -S -o {asm_file} {src_file}",
                shell=True, check=True, capture_output=True, text=True
            )
            result["assembly"] = asm_file.read_text()
            
            subprocess.run(
                f"clang {flags} -S -emit-llvm -o {ir_file} {src_file}",
                shell=True, check=True, capture_output=True, text=True
            )
            result["ir"] = ir_file.read_text()[:2000]
            
            dis_output = subprocess.run(
                f"llvm-objdump -d {obj_file}",
                shell=True, capture_output=True, text=True
            )
            result["disassembly"] = dis_output.stdout
            
            result["success"] = True
            
        except subprocess.CalledProcessError as e:
            result["error"] = e.stderr
            
        return result
    
    def create_test_harness(self, func_signature: str, func_code: str, test_code: str) -> str:
        return f"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

{func_code}

{test_code}

int main()
{{
    return run_tests();
}}
"""
    
    def validate_function(self, func_code: str, test_code: str, func_signature: str) -> Tuple[bool, str]:
        harness = self.create_test_harness(func_signature, func_code, test_code)
        test_file = self.work_dir / "test.c"
        test_file.write_text(harness)
        test_bin = self.work_dir / "test"
        
        try:
            subprocess.run(
                f"clang -fsanitize=address,undefined -o {test_bin} {test_file}",
                shell=True, check=True, capture_output=True, text=True
            )
            
            result = subprocess.run(
                str(test_bin),
                capture_output=True, text=True, timeout=5
            )
            
            if result.returncode != 0:
                return False, f"Test failed: {result.stderr}"
                
            return True, "All tests passed"
            
        except subprocess.CalledProcessError as e:
            return False, f"Compilation failed: {e.stderr}"
        except subprocess.TimeoutExpired:
            return False, "Test timeout"
        except Exception as e:
            return False, str(e)
    
    def benchmark_function(self, func_code: str, bench_code: str, func_signature: str, iterations: int = 1000000) -> float:
        bench_harness = f"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

{func_code}

{bench_code}

int main()
{{
    struct timespec start, end;
    
    setup_benchmark();
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    for (int i = 0; i < {iterations}; i++)
    {{
        run_benchmark();
    }}
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("%f\\n", elapsed);
    
    return 0;
}}
"""
        
        bench_file = self.work_dir / "bench.c"
        bench_file.write_text(bench_harness)
        bench_bin = self.work_dir / "bench"
        
        try:
            subprocess.run(
                f"clang -O3 -o {bench_bin} {bench_file}",
                shell=True, check=True, capture_output=True, text=True
            )
            
            times = []
            for _ in range(5):
                result = subprocess.run(
                    str(bench_bin),
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    times.append(float(result.stdout.strip()))
            
            if times:
                return statistics.median(times)
            else:
                return float('inf')
                
        except Exception:
            return float('inf')
    
    def create_evidence_bundle(self, func_code: str, func_signature: str, compilation_result: Dict, baseline_time: float) -> Dict:
        return {
            "function_signature": func_signature,
            "source_code": func_code,
            "assembly": compilation_result.get("assembly", "")[:3000],
            "ir_snippet": compilation_result.get("ir", "")[:1500],
            "compiler_flags": compilation_result.get("flags", "-O3"),
            "baseline_time_seconds": baseline_time,
            "platform": "macOS/Linux with clang",
            "constraints": [
                "No undefined behavior",
                "Exact same outputs for all inputs",
                "Valid C code that compiles with clang",
                "Thread-safe if original is thread-safe"
            ]
        }
    
    def call_performance_agent(self, evidence: Dict) -> Optional[str]:
        prompt = f"""You are a C performance optimization expert. Given the following C function and analysis, suggest an optimized version that runs faster while maintaining exact correctness.

Function signature: {evidence['function_signature']}

Current implementation:
```c
{evidence['source_code']}
```

Assembly output (truncated):
```
{evidence['assembly'][:1500]}
```

Current median runtime: {evidence['baseline_time_seconds']:.6f} seconds for 1M iterations

Constraints:
- {chr(10).join('- ' + c for c in evidence['constraints'])}

Provide ONLY the optimized C function code with no explanation, comments, or markdown. The code should be a complete, compilable function."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a C performance optimization expert. Return only valid C code."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            code = response.choices[0].message.content.strip()
            
            if code.startswith("```"):
                lines = code.split('\n')
                code = '\n'.join(lines[1:-1] if lines[-1] == "```" else lines[1:])
            
            return code
            
        except Exception as e:
            print(f"GPT API error: {e}")
            return None
    
    def optimize(self, func_code: str, func_signature: str, test_code: str, bench_code: str, max_iterations: int = 5):
        print(f"Starting optimization of function: {func_signature}")
        print(f"Working directory: {self.work_dir}")
        
        print("\n[1/5] Compiling and analyzing baseline...")
        baseline_result = self.compile_and_analyze(func_code)
        if not baseline_result["success"]:
            print(f"Baseline compilation failed: {baseline_result.get('error', 'Unknown error')}")
            return None
        
        print("[2/5] Validating baseline...")
        valid, msg = self.validate_function(func_code, test_code, func_signature)
        if not valid:
            print(f"Baseline validation failed: {msg}")
            return None
        
        print("[3/5] Benchmarking baseline...")
        baseline_time = self.benchmark_function(func_code, bench_code, func_signature)
        print(f"Baseline time: {baseline_time:.6f}s")
        
        best_code = func_code
        best_time = baseline_time
        
        for iteration in range(max_iterations):
            print(f"\n[Iteration {iteration + 1}/{max_iterations}]")
            
            evidence = self.create_evidence_bundle(best_code, func_signature, baseline_result, best_time)
            
            print("  Calling GPT-4 for optimization suggestions...")
            candidate = self.call_performance_agent(evidence)
            
            if not candidate:
                print("  No candidate received")
                continue
            
            print("  Compiling candidate...")
            candidate_result = self.compile_and_analyze(candidate)
            if not candidate_result["success"]:
                print(f"  Candidate compilation failed")
                continue
            
            print("  Validating candidate...")
            valid, msg = self.validate_function(candidate, test_code, func_signature)
            if not valid:
                print(f"  Candidate validation failed: {msg}")
                continue
            
            print("  Benchmarking candidate...")
            candidate_time = self.benchmark_function(candidate, bench_code, func_signature)
            print(f"  Candidate time: {candidate_time:.6f}s")
            
            speedup = best_time / candidate_time if candidate_time > 0 else 0
            print(f"  Speedup: {speedup:.2f}x")
            
            if candidate_time < best_time * 0.95:
                print(f"  ✓ Accepted! New best time: {candidate_time:.6f}s")
                best_code = candidate
                best_time = candidate_time
                baseline_result = candidate_result
                
                self.iterations.append({
                    "iteration": iteration + 1,
                    "code": candidate,
                    "time": candidate_time,
                    "speedup": baseline_time / candidate_time
                })
            else:
                print(f"  ✗ Not faster enough (need >5% improvement)")
        
        return {
            "original_code": func_code,
            "optimized_code": best_code,
            "original_time": baseline_time,
            "optimized_time": best_time,
            "speedup": baseline_time / best_time if best_time > 0 else 1.0,
            "iterations": self.iterations
        }

def main():
    parser = argparse.ArgumentParser(description="AI-powered C function optimizer")
    parser.add_argument("--func", required=True, help="Path to C function file")
    parser.add_argument("--sig", required=True, help="Function signature")
    parser.add_argument("--test", required=True, help="Path to test code file")
    parser.add_argument("--bench", required=True, help="Path to benchmark code file")
    parser.add_argument("--iterations", type=int, default=5, help="Max optimization iterations")
    parser.add_argument("--output", help="Output file for optimized code")
    
    args = parser.parse_args()
    
    func_code = Path(args.func).read_text()
    test_code = Path(args.test).read_text()
    bench_code = Path(args.bench).read_text()
    
    optimizer = COptimizer()
    
    result = optimizer.optimize(
        func_code=func_code,
        func_signature=args.sig,
        test_code=test_code,
        bench_code=bench_code,
        max_iterations=args.iterations
    )
    
    if result:
        print("\n" + "="*60)
        print("OPTIMIZATION COMPLETE")
        print("="*60)
        print(f"Original time: {result['original_time']:.6f}s")
        print(f"Optimized time: {result['optimized_time']:.6f}s")
        print(f"Speedup: {result['speedup']:.2f}x")
        
        if args.output:
            Path(args.output).write_text(result['optimized_code'])
            print(f"\nOptimized code saved to: {args.output}")
        else:
            print("\nOptimized code:")
            print(result['optimized_code'])
    else:
        print("\nOptimization failed")
        sys.exit(1)

if __name__ == "__main__":
    main()