#!/usr/bin/env python3
"""One-off helper: fetch the Bangladeshi Bangla VITS checkpoint and the Chatterbox Multilingual
ONNX export from Hugging Face, strip the VITS weights (no torch needed) and render Arabic test
samples. Everything that must go back to the main session is written to ./out (small files only).

usage: cd <staging checkout> && python3 helper/helper_run.py
"""
import hashlib, json, os, shutil, subprocess, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), 'out')
WORK = os.path.expanduser('~/helper_work')
BN_REPO = 'EMTIAZZ/bangladeshi-bangla-tts-vits'
CB_REPO = 'Masterx/chatterbox-multilingual-v3-ONNX'
CB_FILES = ['default_voice.wav', 'tokenizer.json',
            'onnx/speech_encoder.onnx', 'onnx/speech_encoder.onnx_data',
            'onnx/embed_tokens.onnx', 'onnx/embed_tokens.onnx_data',
            'onnx/language_model_q4.onnx', 'onnx/language_model_q4.onnx_data',
            'onnx/conditional_decoder.onnx', 'onnx/conditional_decoder.onnx_data']
BN_FILES = ['config.json', 'model_info.json', 'README.md', 'inference.py', 'requirements.txt', 'pytorch_model.pth']
RESULT = {'started': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'steps': []}


def log(*a):
    msg = ' '.join(str(x) for x in a)
    print(msg, flush=True)
    RESULT['steps'].append(msg[:500])


def save():
    os.makedirs(OUT, exist_ok=True)
    json.dump(RESULT, open(os.path.join(OUT, 'RESULT.json'), 'w'), indent=1, ensure_ascii=False)


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def revision(repo):
    with urllib.request.urlopen(f'https://huggingface.co/api/models/{repo}', timeout=60) as r:
        return json.load(r)['sha']


def fetch(repo, rev, name, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return
    url = f'https://huggingface.co/{repo}/resolve/{rev}/{name}'
    t0 = time.time()
    subprocess.run(['curl', '-sSfL', '--retry', '3', '-o', dest + '.tmp', url], check=True)
    os.replace(dest + '.tmp', dest)
    log(f'downloaded {repo}/{name} {os.path.getsize(dest)/1e6:.1f} MB in {time.time()-t0:.0f}s')


def bangla():
    rev = revision(BN_REPO)
    d = os.path.join(WORK, 'bn_bd')
    for f in BN_FILES:
        fetch(BN_REPO, rev, f, os.path.join(d, f))
    od = os.path.join(OUT, 'bn_bd')
    os.makedirs(od, exist_ok=True)
    subprocess.run([sys.executable, '-I', os.path.join(HERE, 'strip_pth.py'), os.path.join(d, 'pytorch_model.pth'), od], check=True)
    for f in BN_FILES[:-1]:
        shutil.copy(os.path.join(d, f), od)
    RESULT['bangla'] = {'repo': BN_REPO, 'revision': rev,
                        'files': {f: sha(os.path.join(d, f)) for f in BN_FILES}}
    log('bangla ok, revision', rev)


def arabic():
    try:
        import onnxruntime, tokenizers, soundfile  # noqa
    except ImportError:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--break-system-packages',
                        'onnxruntime', 'tokenizers', 'soundfile'], check=True)
    rev = revision(CB_REPO)
    md = os.path.join(WORK, 'chatterbox')
    for f in CB_FILES:
        fetch(CB_REPO, rev, f, os.path.join(md, f))
    RESULT['arabic_model'] = {'repo': CB_REPO, 'revision': rev,
                              'files': {f: {'sha256': sha(os.path.join(md, f)), 'bytes': os.path.getsize(os.path.join(md, f))} for f in CB_FILES}}
    save()
    sys.path.insert(0, HERE)
    import soundfile as sf
    from cb_ar import ArabicVoice, SR
    tests = json.load(open(os.path.join(HERE, 'ar_tests.json'), encoding='utf-8'))
    refs = [('default', os.path.join(md, 'default_voice.wav')), ('male', os.path.join(HERE, 'ref_male_24k.wav'))]
    sd = os.path.join(OUT, 'ar_samples')
    os.makedirs(sd, exist_ok=True)
    RESULT['arabic_samples'] = []
    shutil.copy(os.path.join(md, 'default_voice.wav'), os.path.join(sd, 'ref_default_voice.wav'))
    for rname, rpath in refs:
        v = ArabicVoice(md, rpath, log=log)
        for t in tests:
            for tash in ([True, False] if t is tests[0] else [True]):
                t0 = time.time()
                w, info = v.say(t['text'], tashkeel=tash)
                fn = f"{rname}_{t['id']}{'' if tash else '_no-tashkeel'}.wav"
                sf.write(os.path.join(sd, fn), w, SR)
                info.update(file=fn, ref=rname, text=t['text'], tashkeel=tash, wall_s=round(time.time() - t0, 1))
                RESULT['arabic_samples'].append(info)
                log('sample', json.dumps(info, ensure_ascii=False))
                save()


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    for name, fn in [('bangla', bangla), ('arabic', arabic)]:
        try:
            fn()
        except Exception as ex:
            import traceback
            log(f'ERROR in {name}: {ex!r}')
            RESULT.setdefault('errors', []).append({name: traceback.format_exc()[-3000:]})
        save()
    RESULT['finished'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    save()
    print('HELPER DONE', 'errors' in RESULT)
