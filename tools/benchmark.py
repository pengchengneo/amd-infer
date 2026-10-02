"""Serial matched HIP / pinned Windows Vulkan benchmark; never promote a config."""
import argparse
import array
import datetime
import json
import math
import re
from pathlib import Path
import statistics
import subprocess
import sys
from run_engine import ROOT, encode, fresh_output, project_gpu_lock, read_utf8_exact, resolve_runtime, sha256, write_json

MODEL_SHA256 = '40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199'


def windows_path(path):
    return subprocess.check_output(['wslpath', '-w', str(Path(path).resolve())], text=True).strip()


def rates(directory, native):
    result = {}
    pattern = '*.greedy.trial*.metrics.json' if native else '*.ops0.trial*.metrics.json'
    for path in sorted(directory.glob(pattern)):
        metrics = json.loads(path.read_text(encoding='utf-8-sig'))
        name = path.name.split('.greedy.' if native else '.ops0.')[0]
        if metrics['output_tokens'] != 64 or metrics['decode_intervals'] != 63:
            raise ValueError('Unexpected timing protocol.')
        rate = metrics['decode_tokens_per_second']
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError('Invalid measured rate.')
        row = result.setdefault(name, {'trials': []})
        row['trials'].append({'compute_tok_s': rate,
                              'selection_wall_tok_s': 63 / metrics['decode_wall_seconds'],
                              'metrics': metrics})
    for row in result.values():
        if len(row['trials']) != 3:
            raise ValueError('Each case requires all three completed trials.')
        values = [v['compute_tok_s'] for v in row['trials']]
        row.update({'median_tok_s': statistics.median(values), 'min_tok_s': min(values),
                    'max_tok_s': max(values), 'spread_fraction': max(values) / min(values) - 1})
    if not result:
        raise ValueError('No completed metrics; do not summarize an incomplete run.')
    return result


def native_fp32_prompt_check(hip, native, case):
    hprefix, nprefix = hip / (case + '.ops0.trial0'), native / (case + '.greedy.trial0')
    suffix = '.prompt.tokens.json'
    if json.loads(Path(str(hprefix) + suffix).read_text()) != json.loads(Path(str(nprefix) + suffix).read_text()):
        raise ValueError('Reference prompt tokenization differs.')
    def read_values(prefix):
        values = array.array('f'); values.frombytes(Path(str(prefix) + '.prompt.logits.f32').read_bytes())
        if sys.byteorder != 'little': values.byteswap()
        if len(values) != 248320 or not all(map(math.isfinite, values)):
            raise ValueError('Expected finite full-vocabulary prompt logits.')
        return values
    left, right = read_values(hprefix), read_values(nprefix)
    error = max(abs(a - b) for a, b in zip(left, right))
    generated_same = json.loads(Path(str(hprefix) + '.generated.json').read_text()) == json.loads(Path(str(nprefix) + '.generated.json').read_text())
    return {'prompt_max_abs': error, 'original_gate': .001, 'prompt_gate_pass': error <= .001,
            'free_64_generated_ids_identical': generated_same,
            'scope': 'Prompt boundary only; this benchmark does not replace 724/reset/480-sampling/long quality qualification.'}


def summarize(destination, cycles, has_native):
    report = {'cycles': [], 'stable_win_over_both_references': False,
              'quality_suites_rerun': False, 'no_automatic_promotion': True,
              'acceptance_definition': 'All cases/cycles: both HIP brackets have <1% drift, each arm spread<1%, and worst HIP trial exceeds best trial of each native arm; FP32 prompt original gate passes. Performance criterion only, not general quality.',
              'scope': 'All-vocabulary forward+LMhead compute rate; host selection wall rate separate; fixed outputs ignore EOS.'}
    decisions = []
    expected_cases = {p.stem for p in (destination / 'cases').glob('*.txt')}
    if not expected_cases:
        raise ValueError('Missing input case snapshot.')
    for cycle in range(cycles):
        base = destination / f'cycle{cycle}'
        before, after = rates(base / 'hip-before/run', False), rates(base / 'hip-after/run', False)
        row = {'hip_before': before, 'hip_after': after, 'cases': {}}
        fp32 = rates(base / 'native-fp32', True) if has_native else None
        default = rates(base / 'native-default', True) if has_native else None
        for arm in [before, after] + ([fp32, default] if has_native else []):
            if set(arm) != expected_cases:
                raise ValueError('Incomplete case coverage; do not rank partial results.')
        if has_native: row.update({'native_fp32': fp32, 'native_default': default})
        for name in before:
            bit_same = all((base / 'hip-before/run' / f'{name}.ops0.trial0{s}').read_bytes() ==
                           (base / bracket / 'run' / f'{name}.ops0.trial{trial}{s}').read_bytes()
                           for bracket in ['hip-before', 'hip-after'] for trial in range(3)
                           for s in ['.prompt.logits.f32', '.last.logits.f32', '.generated.json'])
            drift = after[name]['median_tok_s'] / before[name]['median_tok_s'] - 1
            candidate = {'before_after_drift_fraction': drift, 'before_after_boundaries_and_ids_bit_exact': bit_same}
            if has_native:
                numerical = native_fp32_prompt_check(base / 'hip-before/run', base / 'native-fp32', name)
                minimum = min(before[name]['min_tok_s'], after[name]['min_tok_s'])
                candidate.update({'native_fp32_prompt': numerical,
                                  'worst_hip_over_best_fp32_fraction': minimum / fp32[name]['max_tok_s'] - 1,
                                  'worst_hip_over_best_default_fraction': minimum / default[name]['max_tok_s'] - 1})
                stable = abs(drift) < .01 and bit_same and all(arm[name]['spread_fraction'] < .01 for arm in [before, after, fp32, default])
                passed = stable and numerical['prompt_gate_pass'] and all(minimum > arm[name]['max_tok_s'] for arm in [fp32, default])
                candidate.update({'stable': stable, 'performance_win_over_both': passed}); decisions.append(passed)
            row['cases'][name] = candidate
        # For the default short-long-short cases, show the same request after a
        # 447-token request. This is intra-process drift, separate from reloads.
        for bracket, arm in [('before', before), ('after', after)]:
            if 'a_short' in arm and 'c_short' in arm:
                row[f'{bracket}_short_after_long_drift_fraction'] = arm['c_short']['median_tok_s'] / arm['a_short']['median_tok_s'] - 1
                candidate = base / f'hip-{bracket}/run'
                row[f'{bracket}_same_short_bits'] = all((candidate / f'a_short.ops0.trial0{s}').read_bytes() == (candidate / f'c_short.ops0.trial{trial}{s}').read_bytes() for trial in range(3) for s in ['.prompt.logits.f32', '.last.logits.f32', '.generated.json'])
        if has_native:
            for name, arm in [('fp32', fp32), ('default', default)]:
                if 'a_short' in arm and 'c_short' in arm:
                    row[f'native_{name}_short_after_long_drift_fraction'] = arm['c_short']['median_tok_s'] / arm['a_short']['median_tok_s'] - 1
                    directory = base / f'native-{name}'
                    row[f'native_{name}_same_short_bits'] = all((directory / f'a_short.greedy.trial0{s}').read_bytes() == (directory / f'c_short.greedy.trial{trial}{s}').read_bytes() for trial in range(3) for s in ['.prompt.logits.f32', '.last.logits.f32', '.generated.json'])
        report['cycles'].append(row)
    report['all_completed_windows_beat_both_references'] = bool(decisions) and all(decisions)
    report['stable_win_over_both_references'] = cycles == 2 and bool(decisions) and all(decisions)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', type=Path, default=ROOT / 'fixtures/benchmark')
    parser.add_argument('--runtime-config', type=Path)
    parser.add_argument('--cycles', type=int, choices=[1, 2], default=2)
    parser.add_argument('--native-runtime', type=Path, help='WSL path to pinned b11284 Windows DLL directory')
    parser.add_argument('--native-manifest', type=Path, help='Known launch.json containing pinned DLL runtime_hashes')
    parser.add_argument('--powershell', default='pwsh.exe', help='Windows PowerShell 7 executable accessible from WSL')
    parser.add_argument('--dry-run', action='store_true', help='Hash/tokenize/preflight only; no GPU workload or result directory')
    args = parser.parse_args()
    try:
        if args.native_manifest and not args.native_runtime:
            raise ValueError('--native-manifest requires --native-runtime.')
        destination = fresh_output(args.output)
        args.model, args.tokenizer = args.model.resolve(strict=True), args.tokenizer.resolve(strict=True)
        model_hash = sha256(args.model)
        if model_hash != MODEL_SHA256:
            raise ValueError('This benchmark protocol requires the recorded IQ4_XS model hash.')
        runtime, env, evidence = resolve_runtime(args.runtime_config)
        cases = args.cases.resolve(strict=True)
        paths = sorted(cases.glob('*.txt'))
        if not paths: raise ValueError('No UTF-8 .txt cases.')
        case_info = {}
        for path in paths:
            ids = encode(args.tokenizer, read_utf8_exact(path))
            if not ids or len(ids) + 64 > 512:
                raise ValueError(f'Case exceeds 512 context: {path}, {len(ids)}+64')
            case_info[path.name] = {'sha256': sha256(path), 'prompt_tokens': len(ids)}
        if args.native_runtime:
            args.native_runtime = args.native_runtime.resolve(strict=True)
            if not (args.native_runtime / 'llama.dll').is_file(): raise ValueError('Expected Windows llama.dll.')
            native_hashes = {path.name: sha256(path) for path in sorted(args.native_runtime.glob('*.dll'))}
            if args.native_manifest:
                expected = json.loads(args.native_manifest.read_text(encoding='utf-8-sig'))['runtime_hashes']
                if any(native_hashes.get(name) != value for name, value in expected.items()):
                    raise ValueError('Pinned native DLL manifest mismatch.')
        else: native_hashes = None
        env.update({'AMD_INFER_PRECONDITION_TOKENS': '256', 'AMD_INFER_CASE_WARMUP': '1',
                    'AMD_INFER_EVAL_REPEATS': '3', 'AMD_INFER_EVAL_MODES': '0',
                    'AMD_INFER_OUTPUT_TOKENS': '64', 'AMD_INFER_CONTEXT_CAPACITY': '512',
                    'AMD_INFER_PARTIAL_RESIDENCY': '0'})
        evidence.update({'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                         'model_sha256': model_hash, 'tokenizer_sha256': sha256(args.tokenizer),
                         'cases': case_info, 'cycles': args.cycles, 'native_runtime_hashes': native_hashes,
                         'env': {k:v for k,v in env.items() if k.startswith('AMD_INFER_')},
                         'protocol': {'context':512,'batch':1,'outputs':64,'decode_intervals':63,'warm_tokens':256,'case_warmup':True,'trials_per_arm':3},
                         'order': ([['hip-before','native-fp32','native-default','hip-after'], ['hip-before','native-default','native-fp32','hip-after']][:args.cycles] if args.native_runtime else [['hip-before','hip-after']] * args.cycles),
                         'cold_load_and_warmup_in_headline': False, 'failed_runs_preserved_and_excluded': True})
        if args.dry_run:
            print(json.dumps(evidence, indent=2)); return 0
        with project_gpu_lock():
            destination.mkdir(parents=True, exist_ok=False); write_json(destination / 'launch.json', evidence)
            # Snapshot input files: later editor changes cannot alter a running protocol.
            local_cases = destination / 'cases'; local_cases.mkdir()
            for path in paths: (local_cases / path.name).write_bytes(path.read_bytes())
            for cycle in range(args.cycles):
                base = destination / f'cycle{cycle}'; base.mkdir()
                order = ['hip-before', 'native-fp32', 'native-default', 'hip-after'] if cycle == 0 else ['hip-before', 'native-default', 'native-fp32', 'hip-after']
                if not args.native_runtime: order = ['hip-before', 'hip-after']
                for arm in order:
                    output = base / arm
                    if arm.startswith('hip'):
                        output.mkdir()
                        command = [str(runtime / 'evaluate'), str(args.model), str(args.tokenizer), str(local_cases), str(output / 'run')]
                    else:
                        command = [args.powershell, '-NoProfile', '-File', windows_path(ROOT / 'tools/benchmark_native.ps1'),
                                   '-Mode', arm.removeprefix('native-'), '-Runtime', windows_path(args.native_runtime),
                                   '-Model', windows_path(args.model), '-Cases', windows_path(local_cases), '-Output', windows_path(output)]
                        if args.native_manifest: command += ['-ExpectedManifest', windows_path(args.native_manifest)]
                    write_json(base / (arm + '.launch.json'), {'command':command,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
                    print(f'cycle={cycle} arm={arm}; strictly serial', flush=True)
                    with (base / (arm + '.log')).open('w', encoding='utf-8') as log:
                        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                    write_json(base / (arm + '.exit.json'), {'returncode': result.returncode})
                    cache_valid = True
                    if arm.startswith('hip'):
                        cache = re.findall(r'resident admission packed_bytes=(\d+)', (base / (arm + '.log')).read_text())
                        cache_valid = cache == ['13333954560']
                    if result.returncode or not cache_valid:
                        failure = result.returncode or 2
                        write_json(destination / 'status.json', {'complete':False,'failed_cycle':cycle,'failed_arm':arm,'returncode':failure,'resident_set_valid':cache_valid,'no_performance_claim':True})
                        print(f'Incomplete; preserved logs: {base / (arm + ".log")}'); return failure
            report = summarize(destination, args.cycles, bool(args.native_runtime))
            write_json(destination / 'summary.json', report); write_json(destination / 'status.json', {'complete':True})
        print(f'Complete: {destination}/summary.json; stable win over both references={report["stable_win_over_both_references"]}')
        return 0
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(2, str(error) + '\n')


if __name__ == '__main__':
    raise SystemExit(main())
