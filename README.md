# izhive-media

`main` is temporary hand-off storage for Izhive Shorts uploads (kept empty).

`models` holds the Bangla voice used for the Shorts voiceover: AI4Bharat Indic-TTS Bengali FastPitch + HiFi-GAN
(https://github.com/AI4Bharat/Indic-TTS, MIT licence, see tts/bn/LICENSE-AI4Bharat-Indic-TTS.txt), converted to ONNX.
`bn_fastpitch.onnx` is split into parts (`cat bn_fastpitch.onnx.part* > bn_fastpitch.onnx`); checksums in `SHA256SUMS`.
