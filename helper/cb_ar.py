#!/usr/bin/env python3
"""Arabic speech with Chatterbox Multilingual (Resemble AI, MIT) exported to ONNX.

Runtime follows the onnx-community / Masterx V3 export contract:
  speech_encoder(audio_values[1,N]@24k) -> cond_emb, prompt_token, ref_x_vector, prompt_feat
  embed_tokens(input_ids, position_ids, exaggeration) -> inputs_embeds
  language_model(inputs_embeds, attention_mask, past_key_values.{i}.key/value) -> logits, present.*
  conditional_decoder(speech_tokens, speaker_embeddings, speaker_features) -> wav @24k
Greedy decoding with repetition penalty 1.2. Only onnxruntime, tokenizers, numpy, soundfile needed.

usage (CLI): python3 cb_ar.py MODEL_DIR REF_WAV OUT_WAV "arabic text" [--no-tashkeel]
"""
import json, os, re, subprocess, sys, time, unicodedata
import numpy as np

START_SPEECH, STOP_SPEECH = 6561, 6562
SOT, EOT = 255, 0
SR = 24000
TASHKEEL = re.compile('[ؐ-ًؚ-ٰٟۖ-ۭ]')


def _load_audio_24k(path, max_s=10.0):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-ac', '1', '-ar', str(SR), '-t', str(max_s),
                          '-f', 'f32le', '-'], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def _feed(sess, values):
    """Map our values onto the session's declared inputs (names + dtypes)."""
    feed = {}
    for i in sess.get_inputs():
        if i.name not in values:
            continue
        v = np.asarray(values[i.name])
        want = {'tensor(float)': np.float32, 'tensor(float16)': np.float16, 'tensor(int64)': np.int64,
                'tensor(int32)': np.int32}.get(i.type)
        if want is not None and v.dtype != want:
            v = v.astype(want)
        rank = len(i.shape)
        while v.ndim < rank:
            v = v[None]
        while v.ndim > rank and v.shape[0] == 1:
            v = v[0]
        feed[i.name] = v
    return feed


class ArabicVoice:
    def __init__(self, model_dir, ref_wav, lm='language_model_q4', threads=4, log=print):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        self.log = log
        so = ort.SessionOptions(); so.intra_op_num_threads = threads
        P = ['CPUExecutionProvider']
        od = os.path.join(model_dir, 'onnx')
        t0 = time.time()
        self.enc = ort.InferenceSession(f'{od}/speech_encoder.onnx', so, providers=P)
        self.emb = ort.InferenceSession(f'{od}/embed_tokens.onnx', so, providers=P)
        self.lm = ort.InferenceSession(f'{od}/{lm}.onnx', so, providers=P)
        self.dec = ort.InferenceSession(f'{od}/conditional_decoder.onnx', so, providers=P)
        self.tok = Tokenizer.from_file(os.path.join(model_dir, 'tokenizer.json'))
        log(f'[cb] sessions loaded in {time.time()-t0:.1f}s')
        for s, n in [(self.enc, 'enc'), (self.emb, 'emb'), (self.dec, 'dec')]:
            log(f'[cb] {n} in:', [(i.name, i.type, i.shape) for i in s.get_inputs()],
                'out:', [o.name for o in s.get_outputs()])
        li = self.lm.get_inputs()
        log('[cb] lm in (first 4):', [(i.name, i.type, i.shape) for i in li[:4]], 'n_inputs', len(li))
        self.pkv = [i for i in li if i.name.startswith('past_key_values')]
        self.kv_dtype = np.float16 if self.pkv and self.pkv[0].type == 'tensor(float16)' else np.float32
        out = self.enc.run(None, _feed(self.enc, {'audio_values': _load_audio_24k(ref_wav)[None]}))
        names = [o.name for o in self.enc.get_outputs()]
        self.cond = dict(zip(names, out))
        log('[cb] encoder outputs:', {k: v.shape for k, v in self.cond.items()})

    def ids(self, text):
        text = unicodedata.normalize('NFKD', f'[ar]{text}')
        enc = self.tok.encode(text)
        ids = list(enc.ids)
        unk = [t for t, i in zip(enc.tokens, enc.ids) if t in ('[UNK]', '<unk>')]
        if ids[-1] != START_SPEECH:
            if ids[0] != SOT:
                ids = [SOT] + ids
            if ids[-1] != EOT:
                ids = ids + [EOT]
            ids = ids + [START_SPEECH]
        return np.array([ids], dtype=np.int64), enc.tokens, unk

    def say(self, text, exaggeration=0.5, rep_penalty=1.2, max_new=None, tashkeel=True):
        if not tashkeel:
            text = TASHKEEL.sub('', text)
        input_ids, toks, unk = self.ids(text)
        self.log(f'[cb] {input_ids.shape[1]} text tokens, unk={len(unk)}; tokens: {toks[:60]}')
        if max_new is None:
            max_new = int(min(1000, 60 + 18 * len(TASHKEEL.sub('', text))))
        pos = np.where(input_ids >= START_SPEECH, 0, np.arange(input_ids.shape[1])[None] - 1).astype(np.int64)
        ex = np.array([exaggeration], dtype=np.float32)
        past = {i.name: np.zeros((1, i.shape[1] if isinstance(i.shape[1], int) else 16, 0,
                                  i.shape[3] if isinstance(i.shape[3], int) else 64), dtype=self.kv_dtype)
                for i in self.pkv}
        lm_out = [o.name for o in self.lm.get_outputs()]
        gen = [START_SPEECH]
        t0 = time.time()
        am = None
        for step in range(max_new):
            e = self.emb.run(None, _feed(self.emb, {'input_ids': input_ids, 'position_ids': pos, 'exaggeration': ex}))[0]
            if step == 0:
                e = np.concatenate([self.cond['cond_emb'].astype(e.dtype), e], axis=1)
                am = np.ones((1, e.shape[1]), dtype=np.int64)
            feed = {'inputs_embeds': e, 'attention_mask': am}
            feed.update(past)
            outs = self.lm.run(None, _feed(self.lm, feed))
            logits = outs[0][:, -1, :].astype(np.float32).copy()
            for t in set(gen):
                v = logits[0, t]
                logits[0, t] = v / rep_penalty if v > 0 else v * rep_penalty
            nxt = int(np.argmax(logits[0]))
            gen.append(nxt)
            if nxt == STOP_SPEECH:
                break
            for name, val in zip(lm_out[1:], outs[1:]):
                past[name.replace('present', 'past_key_values')] = val
            input_ids = np.array([[nxt]], dtype=np.int64)
            pos = np.array([[step + 1]], dtype=np.int64)
            am = np.concatenate([am, np.ones((1, 1), dtype=np.int64)], axis=1)
        stopped = gen[-1] == STOP_SPEECH
        speech = [t for t in gen[1:] if t < START_SPEECH]
        self.log(f'[cb] generated {len(speech)} speech tokens in {time.time()-t0:.1f}s, stopped={stopped}')
        st = np.concatenate([self.cond['prompt_token'].astype(np.int64), np.array([speech], dtype=np.int64)], axis=1)
        wav = self.dec.run(None, _feed(self.dec, {'speech_tokens': st,
                                                  'speaker_embeddings': self.cond['ref_x_vector'],
                                                  'speaker_features': self.cond['prompt_feat']}))[0]
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        n = len(speech)
        if n > 1 and len(wav) > (n - 1) * 960:
            wav = wav[:(n - 1) * 960]
        peak = float(np.max(np.abs(wav))) or 1.0
        return (0.9 * wav / peak).astype(np.float32), dict(tokens=n, stopped=stopped, unk=len(unk),
                                                          seconds=round(len(wav) / SR, 2))


if __name__ == '__main__':
    import soundfile as sf
    md, ref, out, text = sys.argv[1:5]
    v = ArabicVoice(md, ref)
    w, info = v.say(text, tashkeel='--no-tashkeel' not in sys.argv)
    sf.write(out, w, SR)
    print(json.dumps(info))
