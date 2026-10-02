"""Run the existing B=1 HIP engine with explicit prompts and recorded settings."""
import argparse
import contextlib
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def fresh_output(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT / 'results') or path == ROOT / 'results':
        raise ValueError('Output must be a new directory below this repository\'s results/.')
    if path.exists():
        raise ValueError(f'Preserve existing output: {path}')
    return path


def resolve_runtime(config_path=None):
    config_path = Path(config_path or ROOT / 'configs/rx9070xt-iq4-xs-fp32.json').resolve()
    config = json.loads(config_path.read_text(encoding='utf-8'))
    env = {k: v for k, v in os.environ.items()
           if not k.startswith('AMD_INFER_') and k not in ('LD_LIBRARY_PATH', 'LD_PRELOAD')}
    env.update(config['env'])
    if env.get('AMD_INFER_ENFORCE_RESERVE') != '1' or env.get('AMD_INFER_STABLE_RMS') != '1':
        raise ValueError('The reserve gate and stable RMS must remain enabled.')
    if 'runtime_hashes' in config:
        if not config.get('validated'):
            raise ValueError('Frozen runtime config has not passed its quality gates.')
        hashes = config['runtime_hashes']
        for relative, expected in hashes.items():
            if sha256(ROOT / relative) != expected:
                raise ValueError(f'Frozen runtime changed: {relative}')
        evaluate = next(ROOT / p for p in hashes if Path(p).name == 'evaluate')
        runtime = evaluate.parent
        library = runtime / 'libamdinfer_kernels.so'
    else:
        runtime = ROOT / 'target/release'
        evaluate = runtime / 'evaluate'
        if not evaluate.is_file():
            raise ValueError('Build first: cargo build --release --features hip,tokenizer --bins')
        linked = subprocess.check_output(['ldd', str(evaluate)], env=env, text=True)
        lines = [line for line in linked.splitlines() if 'libamdinfer_kernels.so =>' in line]
        if len(lines) != 1 or 'not found' in lines[0]:
            raise ValueError('The release binary must link the HIP kernel library.')
        library = Path(lines[0].split('=>')[1].split('(')[0].strip()).resolve()
        hashes = {str(p.relative_to(ROOT)): sha256(p)
                  for p in [evaluate, runtime / 'generate', library]}
    env['LD_LIBRARY_PATH'] = str(library.parent)
    env.update({'AMD_INFER_GPU_EXCLUSIVE': '1', 'AMD_INFER_PROFILE_HOST': '0',
                'AMD_INFER_PROFILE_KERNEL': '0', 'AMD_INFER_PROFILE_ATTN_PARTS': '0',
                'AMD_INFER_PROFILE_GDN_PARTS': '0', 'AMD_INFER_MEMORY_TRACE': '0',
                'AMD_INFER_SAVE_TRAJECTORY': '0', 'AMD_INFER_LOCAL_GRAPHS': '0'})
    return runtime, env, {'config': str(config_path), 'config_sha256': sha256(config_path),
                          'runtime_hashes': hashes, 'frozen_runtime': 'runtime_hashes' in config}


def read_utf8_exact(path):
    with Path(path).open('r', encoding='utf-8', newline='') as source:
        return source.read()


def encode(tokenizer, text):
    binary = ROOT / 'target/release/tokenize'
    if not binary.is_file():
        raise ValueError('Build the CPU tokenizer first with --features hip,tokenizer --bins.')
    with tempfile.TemporaryDirectory(prefix='amd-infer-tokenize-') as directory:
        source, output = Path(directory) / 'input.txt', Path(directory) / 'ids.json'
        source.write_text(text, encoding='utf-8')
        subprocess.run([str(binary), 'encode', str(tokenizer), str(source), str(output)],
                       check=True, stdout=subprocess.DEVNULL)
        return json.loads(output.read_text())


def render_prompt(args):
    text = read_utf8_exact(args.prompt_file)
    if args.template == 'raw':
        if args.system_file or args.tokenizer_config or args.thinking != 'off':
            raise ValueError('System/template/thinking options require --template chat.')
        return text, None
    if not args.tokenizer_config:
        raise ValueError('Chat requires the local official --tokenizer-config JSON.')
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    config = json.loads(args.tokenizer_config.read_text(encoding='utf-8'))
    template = config.get('chat_template')
    if not isinstance(template, str):
        raise ValueError('Expected one text chat_template in tokenizer_config.json.')
    messages = []
    if args.system_file:
        messages.append({'role': 'system', 'content': read_utf8_exact(args.system_file)})
    messages.append({'role': 'user', 'content': text})
    def raise_exception(message):
        raise ValueError(message)
    renderer = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    renderer.globals['raise_exception'] = raise_exception
    prompt = renderer.from_string(template).render(
        messages=messages, add_generation_prompt=True, enable_thinking=args.thinking != 'off',
        reasoning_effort='xhigh' if args.thinking == 'off' else args.thinking)
    return prompt, {'messages': messages, 'thinking': args.thinking,
                    'tokenizer_config_sha256': sha256(args.tokenizer_config)}


@contextlib.contextmanager
def project_gpu_lock():
    import fcntl
    (ROOT / 'results').mkdir(exist_ok=True)
    with (ROOT / 'results/.gpu-run.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Another run-engine/benchmark process holds the project GPU lock.')
        lock.seek(0); lock.truncate(); lock.write(str(os.getpid()) + '\n'); lock.flush()
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--prompt-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New result directory')
    parser.add_argument('--max-new-tokens', type=int, default=64)
    parser.add_argument('--context', type=int, choices=[128, 256, 512], default=512)
    parser.add_argument('--runtime-config', type=Path, help='Optional validated frozen runtime config')
    parser.add_argument('--template', choices=['raw', 'chat'], default='raw')
    parser.add_argument('--tokenizer-config', type=Path)
    parser.add_argument('--system-file', type=Path)
    parser.add_argument('--thinking', choices=['off', 'low', 'medium', 'xhigh'], default='off')
    parser.add_argument('--verify-model', action='store_true', help='Record a full model SHA256')
    parser.add_argument('--dry-run', action='store_true', help='CPU preflight only; create no result directory')
    args = parser.parse_args()
    try:
        if not 1 <= args.max_new_tokens <= 512:
            raise ValueError('--max-new-tokens must be in 1..512.')
        args.model, args.tokenizer = args.model.resolve(strict=True), args.tokenizer.resolve(strict=True)
        destination = fresh_output(args.output)
        runtime, env, evidence = resolve_runtime(args.runtime_config)
        prompt, template = render_prompt(args)
        ids = encode(args.tokenizer, prompt)
        if not ids or len(ids) + args.max_new_tokens > args.context:
            raise ValueError(f'Prompt {len(ids)} + requested output {args.max_new_tokens} exceeds context {args.context}.')
        env['AMD_INFER_CONTEXT_CAPACITY'] = str(args.context)
        command = [str(runtime / 'generate'), str(args.model), str(args.tokenizer),
                   str(destination / 'prompt.txt'), str(destination / 'output.txt'), str(args.max_new_tokens)]
        evidence.update({'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                         'command': command, 'env': {k:v for k,v in env.items() if k.startswith('AMD_INFER_')},
                         'model_bytes': args.model.stat().st_size, 'model_sha256': sha256(args.model) if args.verify_model else None,
                         'tokenizer_sha256': sha256(args.tokenizer), 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                         'prompt_tokens': len(ids), 'context': args.context, 'template': template,
                         'scope': 'Real greedy generate CLI; stops on EOS or limit; no warmup; streaming file I/O is timed. Use benchmark.py for steady-state rates.',
                         'project_lock_scope': 'Only these two launchers; GPU_EXCLUSIVE is an assertion, not a system GPU reservation.'})
        if args.dry_run:
            print(json.dumps(evidence, ensure_ascii=False, indent=2)); return 0
        with project_gpu_lock():
            destination.mkdir(parents=True, exist_ok=False)
            (destination / 'prompt.txt').write_text(prompt, encoding='utf-8')
            write_json(destination / 'prompt.tokens.json', ids)
            write_json(destination / 'launch.json', evidence)
            with (destination / 'run.log').open('w', encoding='utf-8') as log:
                result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            write_json(destination / 'exit.json', {'returncode': result.returncode})
        if result.returncode:
            print(f'Engine failed; evidence preserved in {destination}/run.log')
        else:
            print((destination / 'output.txt').read_text(encoding='utf-8'))
            print(f'Files and timing scope: {destination}')
        return result.returncode
    except (ValueError, OSError, subprocess.CalledProcessError, ImportError) as error:
        parser.exit(2, str(error) + '\n')


if __name__ == '__main__':
    raise SystemExit(main())
