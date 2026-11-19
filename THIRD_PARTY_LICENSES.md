# Third-party licenses and model notes

This file distinguishes Python package source, native libraries, command-line programs, pretrained weights, and training-data terms. A package license does not automatically cover its weights or datasets. Versions in `pyproject.toml` and `requirements.txt` are ranges, not a lockfile; audit the exact installed distributions and native libraries before redistributing a build.

## Required Python packages

The core pipeline uses NumPy (modified BSD), Pillow (MIT-CMU/PIL), python-soundfile (BSD 3-Clause), and jsonschema (MIT). The optional test/resource group uses pytest (MIT) and psutil (BSD 3-Clause). SciPy, PyTorch, SpeechBrain, and Silero VAD are optional dependencies in the `neural` extra; OpenCV is not required. Model files are not bundled. Selecting an optional backend may cause its library to download model weights into the configured cache. Keep each dependency's copyright and license notice when redistributing its source or binary distribution. [1] [2] [3] [4] [5] [6] [26] [27]

`python-soundfile` is BSD-licensed, but it calls the native libsndfile library. Some binary wheels may bundle libsndfile, whose license is LGPL; inspect the exact wheel and platform package before redistribution. [4] [25]

## FFmpeg

FFmpeg is an external system dependency, not a Python package bundled here. FFmpeg's default license is LGPL 2.1 or later, while optional GPL components can make a build subject to GPL terms. The license of an installed FFmpeg build depends on its configure options and linked components. This project invokes H.264 encoding through `libx264`; check the package build configuration and applicable codec/patent obligations before redistributing binaries or rendered media. [7]

## Optional speech components (not enabled in the base pipeline)

**Silero VAD.** The official repository's code license is MIT. The package includes JIT and ONNX model files. An official repository discussion answers “MIT” to a question about model-weight licensing, but the reviewed sources do not provide a separate license file tied to each weight file or release. Treat that as the available official statement, not as a per-checkpoint license manifest. Silero VAD detects speech/non-speech intervals; it does not diarize or separate speakers. Its documented interface expects mono audio at 8 or 16 kHz. [8] [9] [10]

**SpeechBrain and speaker embeddings.** SpeechBrain's code is Apache-2.0. The specific `speechbrain/spkrec-ecapa-voxceleb` model card/API lists Apache-2.0 for that model's published files. ECAPA-TDNN provides speaker embeddings; it is not, by itself, a complete diarization system. Review every selected model card and the terms for training data before use. CPU inference is supported by the library, but the cited sources do not promise a minimum CPU specification or a fixed inference speed. [11] [12] [13]

**SepFormer separation.** The specific `speechbrain/sepformer-wsj03mix` checkpoint is intended for three-source WSJ0-3Mix mixtures; do not substitute the two-source `speechbrain/sepformer-wsj02mix` checkpoint. Both model cards/API data list Apache-2.0 for the published model files. The WSJ0 source data is separately governed by LDC terms; the checkpoint license does not replace those terms. The documented input is mono 8 kHz. The benchmark result is not a guarantee of separation quality on conversational recordings, reverberation, noise, or arbitrary three-speaker mixtures. No inference RAM minimum is specified in the reviewed sources. [14] [15] [16] [17] [18]

**Rhubarb Lip Sync.** Rhubarb is an optional external command-line program under MIT. It accepts WAV/Ogg Vorbis and can output JSON mouth cues (among other formats). Its documentation does not establish diarization or separate cues for three overlapping speakers. PocketSphinx is documented for English; the phonetic recognizer is described as language-independent but usually less accurate. No separate pretrained-weight license was identified in the reviewed official sources. [19] [20]

No pretrained weights are bundled, and the base installation does not download them. When a user installs the optional neural dependencies and explicitly selects a model backend, SpeechBrain or Silero may download the requested weights on first use to the local cache. The user must review the exact checkpoint card and terms; library licenses do not automatically govern each weight file or training dataset. The pipeline reports model/backend failures rather than claiming an unavailable neural result. The included ECAPA integration is a basic embedding-clustering stage, not a complete overlap-aware diarization system; SepFormer outputs require listening and manual source-to-character mapping.

## GitHub Actions and public releases

The workflow uses GitHub-hosted Actions runners and publishes rendered output only when the workflow's `publish_release` input is explicitly set to `true`. The reviewed GitHub documentation lists 4 CPU / 16 GB RAM / 14 GB SSD for standard Ubuntu x64 runners on public repositories and 2 CPU / 8 GB RAM / 14 GB SSD for private repositories; the standard job time limit is six hours. Release assets are separate from Actions artifacts: GitHub documents a maximum of 1000 assets per release and a per-file size below 2 GiB, with no total release-size limit in the cited page. These service limits can change; verify current plan and repository settings before relying on them. A Release is public. Do not publish private recordings or production assets there. [21] [22] [23]

The repository workflow sets `contents: write` only on the separate release job. GitHub documents that release upload needs repository contents write permission. Keep token permissions minimal and follow organization-level policy. [24]

## License files

The AI Podcast Studio source and original generated placeholder artwork in this repository use the MIT License in `LICENSE`. Third-party software, native libraries, model weights, datasets, and hosted services remain under their own terms as described above. This file is a technical inventory, not legal advice.

## References

[1]: https://pillow.readthedocs.io/en/stable/about.html "Pillow license"  
[2]: https://numpy.org/about/ "NumPy project and license"  
[3]: https://github.com/scipy/scipy/blob/main/LICENSE.txt "SciPy license"  
[4]: https://github.com/bastibe/python-soundfile/blob/master/LICENSE "python-soundfile license and native dependency notes"  
[5]: https://github.com/python-jsonschema/jsonschema/blob/main/json/LICENSE "jsonschema license"  
[6]: https://docs.pytest.org/en/stable/license.html "pytest license"  
[7]: https://www.ffmpeg.org/legal.html "FFmpeg legal and licensing information"  
[8]: https://github.com/snakers4/silero-vad/blob/master/LICENSE "Silero VAD source license"  
[9]: https://github.com/snakers4/silero-vad/blob/master/src/silero_vad/model.py "Silero VAD model loader"  
[10]: https://github.com/snakers4/silero-vad/discussions/817 "Silero VAD official discussion on model-weight licensing"  
[11]: https://github.com/speechbrain/speechbrain/blob/develop/LICENSE "SpeechBrain source license"  
[12]: https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb "SpeechBrain ECAPA-TDNN model card"  
[13]: https://github.com/speechbrain/speechbrain/blob/develop/recipes/AMI/Diarization/README.md "SpeechBrain AMI diarization recipe"  
[14]: https://huggingface.co/speechbrain/sepformer-wsj03mix "SpeechBrain SepFormer three-source model card"  
[15]: https://huggingface.co/speechbrain/sepformer-wsj02mix "SpeechBrain SepFormer two-source model card"  
[16]: https://github.com/speechbrain/speechbrain/blob/develop/recipes/WSJ0Mix/separation/README.md "SpeechBrain WSJ0Mix separation recipe"  
[17]: https://catalog.ldc.upenn.edu/LDC93S6A "LDC CSR-I WSJ0 catalog and access terms"  
[18]: https://speechbrain.readthedocs.io/en/latest/API/speechbrain.inference.separation.html "SpeechBrain separation API"  
[19]: https://github.com/DanielSWolf/rhubarb-lip-sync/blob/master/LICENSE.md "Rhubarb Lip Sync license"  
[20]: https://github.com/DanielSWolf/rhubarb-lip-sync/blob/master/README.adoc "Rhubarb Lip Sync usage and output formats"  
[21]: https://docs.github.com/en/actions/reference/runners/github-hosted-runners "GitHub-hosted runner specifications"  
[22]: https://docs.github.com/en/actions/reference/limits "GitHub Actions limits"  
[23]: https://docs.github.com/en/rest/releases/assets "GitHub release asset limits"  
[24]: https://docs.github.com/en/actions/tutorials/authenticate-with-github_token "GitHub Actions GITHUB_TOKEN permissions"
[25]: https://python-soundfile.readthedocs.io/en/latest/ "python-soundfile documentation on libsndfile and binary wheels"
[26]: https://github.com/giampaolo/psutil/blob/master/LICENSE "psutil license"
[27]: https://github.com/pytorch/pytorch/blob/main/LICENSE "PyTorch license"
