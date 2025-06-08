# Sound Effects Directory

This directory contains sound effects used to enhance video engagement and storytelling.

## Directory Structure

```
sound_effects/
├── impact/
│   ├── impact.wav
│   ├── hit.wav
│   ├── thud.wav
│   ├── bang.wav
│   └── crash.wav
├── transition/
│   ├── whoosh.wav
│   ├── swoosh.wav
│   ├── zip.wav
│   ├── swish.wav
│   └── wind.wav
├── liquid/
│   ├── splash.wav
│   ├── pour.wav
│   ├── glug.wav
│   ├── water.wav
│   └── liquid.wav
├── mechanical/
│   ├── click.wav
│   ├── pop.wav
│   ├── snap.wav
│   ├── tick.wav
│   └── beep.wav
├── notification/
│   ├── ding.wav
│   ├── chime.wav
│   ├── bell.wav
│   ├── alert.wav
│   └── notification.wav
└── dramatic/
    ├── boom.wav
    ├── thunder.wav
    ├── rumble.wav
    └── tension.wav
```

## Sound Effect Usage

The system automatically maps AI-suggested sound effects to available files:

### Mapping Examples
- **"splash"** → `liquid/splash.wav` or `liquid/water.wav`
- **"whoosh"** → `transition/whoosh.wav` or `transition/swish.wav`
- **"impact"** → `impact/impact.wav` or `impact/hit.wav`
- **"pop"** → `mechanical/pop.wav` or `mechanical/click.wav`

## File Requirements

- **Format**: WAV, MP3, or OGG
- **Duration**: 0.5-3 seconds recommended
- **Quality**: 44.1kHz, 16-bit minimum
- **Volume**: Normalized to prevent clipping
- **Licensing**: Royalty-free for commercial use

## Fallback Strategy

1. **Exact match**: `effect_name.wav`
2. **Category match**: Files in appropriate subdirectory
3. **Partial match**: Files containing effect name in filename
4. **Alternative mapping**: Predefined alternatives (e.g., "click" → "pop")

## Recommended Sources

- **Freesound.org** (Creative Commons)
- **Zapsplat** (Free with attribution)
- **Adobe Audition** (Built-in library)
- **YouTube Audio Library** (Royalty-free)

## Usage in AI Analysis

The enhanced AI prompts will suggest contextual sound effects:

```json
"sound_effects": [
    {"timestamp_seconds": 2.0, "effect_name": "whoosh", "volume": 0.8},
    {"timestamp_seconds": 8.0, "effect_name": "splash", "volume": 0.9},
    {"timestamp_seconds": 15.0, "effect_name": "pop", "volume": 0.7}
]
```

## Integration Notes

- Sound effects are automatically faded in/out (0.05s) for smooth integration
- Volume levels are adjustable per effect (0.1-1.0 range)
- Effects are validated against video duration to prevent overrun
- Multiple format support with automatic discovery