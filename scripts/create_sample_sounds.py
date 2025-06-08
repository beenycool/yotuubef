"""
Create sample sound effects for testing the enhancement system.
"""

import wave
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def create_test_sound(filename, duration=1.0, frequency=440):
    """Create a simple sine wave sound effect"""
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    # Generate sine wave
    wave_data = np.sin(frequency * 2 * np.pi * t) * 0.3
    # Convert to 16-bit integers
    wave_data = (wave_data * 32767).astype(np.int16)
    
    # Write WAV file
    with wave.open(str(filename), 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 2 bytes per sample
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(wave_data.tobytes())


def main():
    """Create sample sound effects"""
    
    # Define sound effects with duration and frequency
    sounds = {
        'sound_effects/transition/swoosh.wav': (0.8, 200),
        'sound_effects/transition/swish.wav': (0.6, 300),
        'sound_effects/impact/hit.wav': (0.5, 100),
        'sound_effects/impact/thud.wav': (0.7, 80),
        'sound_effects/impact/bang.wav': (0.4, 120),
        'sound_effects/impact/crash.wav': (1.0, 90),
        'sound_effects/liquid/pour.wav': (1.5, 150),
        'sound_effects/liquid/glug.wav': (0.4, 120),
        'sound_effects/liquid/water.wav': (1.2, 180),
        'sound_effects/mechanical/pop.wav': (0.3, 800),
        'sound_effects/mechanical/snap.wav': (0.2, 1000),
        'sound_effects/mechanical/tick.wav': (0.1, 1200),
        'sound_effects/mechanical/beep.wav': (0.5, 880),
        'sound_effects/notification/chime.wav': (1.0, 660),
        'sound_effects/notification/bell.wav': (1.2, 880),
        'sound_effects/notification/alert.wav': (0.8, 1000),
        'sound_effects/dramatic/rumble.wav': (2.0, 60),
        'sound_effects/dramatic/tension.wav': (3.0, 40),
        'sound_effects/dramatic/boom.wav': (1.5, 50),
    }
    
    created_count = 0
    
    for filepath, (duration, freq) in sounds.items():
        file_path = Path(filepath)
        
        # Create directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create sound file if it doesn't exist
        if not file_path.exists():
            try:
                create_test_sound(file_path, duration, freq)
                print(f'[OK] Created {filepath}')
                created_count += 1
            except Exception as e:
                print(f'[FAIL] Failed to create {filepath}: {e}')
        else:
            print(f'- {filepath} already exists')
    
    print(f'\nCreated {created_count} new sound effect files.')
    print('Sample sound effects setup complete!')


if __name__ == "__main__":
    main()