# Changelog
<!--
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
-->

## [Unreleased]
<!-- track upcoming changes here; move to new versioned section at release time -->

## [2.0] - 16 July 2021

Changes related to the ZeroSpeech challenge:
 - added support for SpokenCOCO dataset
 - added code to support the use of pretrained features + utility script to extract CPC features
 - refactored tokenization helpers making the tokenizer a global variable of dataset.py
 - changed platalea default config path ~/.platalea -> ~/.config/platalea
 - disabled use of wandb by default in basic.py and transformer.py experiments
 - pinning down pytorch version

Resolves issues #53, #103, #104 and (temporarily) solves #106.

## [1.0] - 9 December 2020

### Added
- Introducing an attention-based encoder-decoder architecture for speech recognition.
- Multitask training with multiple objectives (e.g. cross-modality retrieval and speech transcription) is also possible now.

<!--
### Removed

### Changed
-->

## [0.9] - 20 January 2020

State of the repo before @bhigy's merge leading to version 1.0.

[Unreleased]: https://github.com/spokenlanguage/platalea/compare/v1.0...HEAD
[2.0]: https://github.com/spokenlanguage/platalea/releases/tag/v2.0
[1.0]: https://github.com/spokenlanguage/platalea/releases/tag/v1.0
[0.9]: https://github.com/gchrupala/platalea/releases/tag/v0.9
