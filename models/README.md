# Optional model weights

No model weights are bundled or downloaded by the default pipeline. Optional Silero VAD, SpeechBrain ECAPA embedding/diarization, and SpeechBrain SepFormer 2/3-source backends are available after installing the `neural` extra and selecting them in the project JSON. The model library may download weights on first selected use. Before enabling a backend, review `../THIRD_PARTY_LICENSES.md`, verify the exact checkpoint card and weight license, and keep cached files out of a public repository. A model-library license does not automatically license its pretrained weights or training data.

Current default behavior is explicitly non-neural: energy-threshold VAD, manually supplied speaker intervals, no source separation, and signal-envelope mouth cues. Neural outputs are optional and require review; ECAPA cluster labels and SepFormer source indices are not character identities.
