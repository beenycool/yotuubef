"""
Meme Generator Module for Gen Z Video Enhancement
Adds meme overlays, emojis, and trending text to video frames for viral appeal.
"""

import logging
import random
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

try:
    import numpy as np
    import cv2
    from PIL import Image, ImageDraw, ImageFont
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False
    np = None
    cv2 = None
    Image = None
    ImageDraw = None
    ImageFont = None

from src.config.settings import get_config


class MemeGenerator:
    """
    Generates and applies meme overlays, emojis, and Gen Z text to video frames
    """
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        
        # Check dependencies
        if not PILLOW_AVAILABLE:
            self.logger.warning("⚠️ Pillow not available - meme overlays will be disabled")
            self.available = False
        else:
            self.available = True
        
        # Gen Z meme templates and phrases
        self.meme_phrases = [
            "SUS AF 😂", "NO CAP 💀", "LITERALLY SHAKING RN 😭", "BUSSIN 🔥",
            "FR FR ✨", "SLAPS 👀", "ABSOLUTELY WILD 😱", "POV 💥",
            "PLOT TWIST 🚀", "I CAN'T EVEN 😅", "PEAK CONTENT 🎯", "VIBES 🌟"
        ]
        
        # Emoji combinations for different moods
        self.emoji_combinations = {
            "excited": ["🔥", "💥", "🚀", "✨", "💯"],
            "funny": ["😂", "💀", "😭", "🤣", "😅"],
            "surprised": ["😱", "😨", "😰", "😵", "🤯"],
            "cool": ["😎", "🤙", "💪", "👊", "🔥"],
            "suspicious": ["👀", "🤔", "🧐", "🤨", "😏"]
        }
        
        # Font setup
        self.fonts_dir = Path(self.config.paths.fonts_dir)
        self.default_font = "arial.ttf"  # Fallback font
        
        # Load available fonts
        self.available_fonts = self._load_available_fonts()
        
    def _load_available_fonts(self) -> List[str]:
        """Load available fonts from fonts directory"""
        try:
            if self.fonts_dir.exists():
                font_files = list(self.fonts_dir.glob("*.ttf")) + list(self.fonts_dir.glob("*.otf"))
                return [f.name for f in font_files]
            return []
        except Exception as e:
            self.logger.error(f"Failed to load fonts: {e}")
            return []
    
    def add_meme_overlay(self, frame: np.ndarray, text: str, meme_type: str = "impact", 
                         mood: str = "funny", position: str = "random") -> np.ndarray:
        """
        Add meme overlay to video frame
        
        Args:
            frame: Video frame as numpy array
            text: Text to overlay
            meme_type: Type of meme style ("impact", "classic", "modern")
            mood: Emotional mood for emoji selection
            position: Text position ("top", "bottom", "center", "random")
            
        Returns:
            Frame with meme overlay
        """
        if not self.available or frame is None:
            return frame
        
        try:
            # Convert numpy array to PIL Image
            if len(frame.shape) == 3:
                img = Image.fromarray(frame)
            else:
                # Handle grayscale
                img = Image.fromarray(frame).convert('RGB')
            
            # Add meme text
            img = self._add_meme_text(img, text, meme_type, position)
            
            # Add emojis based on mood
            img = self._add_emojis(img, mood, position)
            
            # Convert back to numpy array
            return np.array(img)
            
        except Exception as e:
            self.logger.error(f"Failed to add meme overlay: {e}")
            return frame
    
    def _add_meme_text(self, img: Image.Image, text: str, meme_type: str, position: str) -> Image.Image:
        """Add meme-style text to image"""
        try:
            draw = ImageDraw.Draw(img)
            
            # Select font and size
            font_size = self._get_font_size(img.size, len(text))
            font = self._get_font(font_size)
            
            # Get text position
            text_pos = self._get_text_position(img.size, position)
            
            # Apply meme styling based on type
            if meme_type == "impact":
                # Classic Impact font style with black outline
                outline_color = "black"
                text_color = "white"
                outline_width = 3
            elif meme_type == "classic":
                # Classic meme style
                outline_color = "black"
                text_color = "white"
                outline_width = 2
            else:  # modern
                # Modern style with gradient-like effect
                outline_color = "black"
                text_color = "#FF6B6B"  # Modern red
                outline_width = 2
            
            # Draw text outline
            for dx in range(-outline_width, outline_width + 1):
                for dy in range(-outline_width, outline_width + 1):
                    if dx != 0 or dy != 0:
                        draw.text((text_pos[0] + dx, text_pos[1] + dy), 
                                text.upper(), fill=outline_color, font=font)
            
            # Draw main text
            draw.text(text_pos, text.upper(), fill=text_color, font=font)
            
            return img
            
        except Exception as e:
            self.logger.error(f"Failed to add meme text: {e}")
            return img
    
    def _add_emojis(self, img: Image.Image, mood: str, position: str) -> Image.Image:
        """Add emojis to image based on mood and position"""
        try:
            draw = ImageDraw.Draw(img)
            
            # Get emojis for mood
            emojis = self.emoji_combinations.get(mood, self.emoji_combinations["funny"])
            
            # Select random emojis
            num_emojis = random.randint(1, 3)
            selected_emojis = random.sample(emojis, min(num_emojis, len(emojis)))
            
            # Get emoji positions
            emoji_positions = self._get_emoji_positions(img.size, position, len(selected_emojis))
            
            # Add emojis
            for i, emoji in enumerate(selected_emojis):
                if i < len(emoji_positions):
                    pos = emoji_positions[i]
                    # For now, we'll use text representation of emojis
                    # In a full implementation, you'd load actual emoji images
                    emoji_font = self._get_font(40)  # Larger font for emojis
                    draw.text(pos, emoji, fill="white", font=emoji_font)
            
            return img
            
        except Exception as e:
            self.logger.error(f"Failed to add emojis: {e}")
            return img
    
    def _get_font(self, size: int) -> ImageFont.FreeTypeFont:
        """Get font with specified size"""
        try:
            # Try to use available fonts first
            if self.available_fonts:
                font_path = self.fonts_dir / self.available_fonts[0]
                return ImageFont.truetype(str(font_path), size)
            else:
                # Fallback to default system font
                return ImageFont.load_default()
        except Exception as e:
            self.logger.warning(f"Failed to load custom font, using default: {e}")
            return ImageFont.load_default()
    
    def _get_font_size(self, img_size: Tuple[int, int], text_length: int) -> int:
        """Calculate appropriate font size based on image size and text length"""
        # Base font size calculation
        base_size = min(img_size[0], img_size[1]) // 20
        
        # Adjust for text length
        if text_length > 20:
            base_size = max(base_size // 2, 20)
        elif text_length > 10:
            base_size = max(base_size // 1.5, 25)
        
        return int(base_size)
    
    def _get_text_position(self, img_size: Tuple[int, int], position: str) -> Tuple[int, int]:
        """Calculate text position on image"""
        width, height = img_size
        
        if position == "top":
            return (width // 2, 20)
        elif position == "bottom":
            return (width // 2, height - 60)
        elif position == "center":
            return (width // 2, height // 2)
        else:  # random
            x = random.randint(50, width - 200)
            y = random.randint(50, height - 100)
            return (x, y)
    
    def _get_emoji_positions(self, img_size: Tuple[int, int], position: str, num_emojis: int) -> List[Tuple[int, int]]:
        """Calculate emoji positions on image"""
        width, height = img_size
        positions = []
        
        if position == "top":
            base_y = 80
            for i in range(num_emojis):
                x = 50 + (i * 60)
                positions.append((x, base_y))
        elif position == "bottom":
            base_y = height - 120
            for i in range(num_emojis):
                x = 50 + (i * 60)
                positions.append((x, base_y))
        else:  # random or center
            for i in range(num_emojis):
                x = random.randint(30, width - 100)
                y = random.randint(30, height - 100)
                positions.append((x, y))
        
        return positions
    
    def generate_random_meme_text(self, content_context: str = "") -> str:
        """Generate random meme text based on context"""
        try:
            # Select random meme phrase
            meme_text = random.choice(self.meme_phrases)
            
            # Add context if provided
            if content_context:
                # Extract key words from context
                words = content_context.split()[:3]
                if words:
                    meme_text = f"{' '.join(words)} {meme_text}"
            
            return meme_text
            
        except Exception as e:
            self.logger.error(f"Failed to generate random meme text: {e}")
            return "ABSOLUTELY WILD 🔥"
    
    def should_add_meme_overlay(self, frame_number: int, analysis: Dict[str, Any] = None) -> bool:
        """Determine if meme overlay should be added to this frame"""
        try:
            # Check if Gen Z mode is enabled
            gen_z_mode = self.config.ai_features.get('gen_z_mode', False)
            if not gen_z_mode:
                return False
            
            # Get meme overlay probability from config
            meme_probability = self.config.gen_z_features.get('meme_overlay_probability', 0.3)
            
            # Check if this is a key moment
            if analysis and analysis.get('is_key_moment', False):
                meme_probability *= 2  # Double probability for key moments
            
            # Random chance based on probability
            return random.random() < meme_probability
            
        except Exception as e:
            self.logger.error(f"Failed to determine meme overlay probability: {e}")
            return False