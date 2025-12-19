"""Configuration for the video generation pipeline."""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class ScriptSegment:
    """A segment of the video script with timing and text."""
    id: str
    start_seconds: int
    end_seconds: int
    text: str
    visual_prompt: str = ""
    # URLs to screenshot for this segment (will be combined with character)
    screenshot_urls: List[str] = field(default_factory=list)
    # Whether this segment needs a screenshot composite
    needs_screenshot: bool = False
    # Direct image URLs to combine with character (no screenshot needed)
    composite_image_urls: List[str] = field(default_factory=list)


# URLs for automatic screenshots
SCREENSHOT_URLS = {
    "wan_playground": "https://fal.ai/models/wan/v2.6/text-to-video/playground",
    "wan_api": "https://fal.ai/models/wan/v2.6/text-to-video/api",
    "fal_home": "https://fal.ai",
    "fal_explore": "https://fal.ai/models",
}

# The full script for the fal Wan 2.6 demo video
# Template segments with {animal} and {style} placeholders
SCRIPT_SEGMENTS_TEMPLATE: List[ScriptSegment] = [
    ScriptSegment(
        id="segment_1",
        start_seconds=0,
        end_seconds=10,
        text="Hey — I'm Lovis Odin, a Creative Engineer at fal. And for this video, I made a tiny character to speak for me: this {animal}. Same voice… different face.",
        visual_prompt="{style} shot of the {animal} character wearing a colorful tropical Hawaiian shirt, introducing itself with a friendly wave, modern studio environment with soft lighting, chill vibes, premium quality",
        needs_screenshot=False  # No screenshot for intro
    ),
    ScriptSegment(
        id="segment_2",
        start_seconds=10,
        end_seconds=22,
        text="This is fal.ai — a developer-first generative media platform. You can run and ship image, video, and audio models through an API, or use the playground to test them directly. Then plug them into real products.",
        visual_prompt="{style} shot of the {animal} character wearing a tropical shirt, presenting the fal.ai homepage interface, gesturing toward the screen showing the fal platform, premium tech environment",
        needs_screenshot=True,
        screenshot_urls=["https://fal.ai"]
    ),
    ScriptSegment(
        id="segment_3",
        start_seconds=22,
        end_seconds=38,
        text="Today I'm showcasing Wan 2.6, available on fal. With text-to-video, you write a prompt, pick a duration — 5, 10, or 15 seconds — choose a clean resolution, and you get cinematic motion.",
        visual_prompt="{style} shot of the {animal} character demonstrating the Wan 2.6 playground interface, pointing at the prompt field and settings, enthusiastic presentation",
        needs_screenshot=True,
        screenshot_urls=["https://fal.ai/models/wan/v2.6/text-to-video/playground"]
    ),
    ScriptSegment(
        id="segment_4",
        start_seconds=38,
        end_seconds=52,
        text="And the best part is multi-lens: one prompt can become a mini sequence — multiple shots, same subject, same vibe — like the model edits the trailer for you.",
        visual_prompt="{style} montage showing the {animal} in different camera angles and environments but maintaining consistent style: wide shot, close-up, profile view, all seamlessly transitioning, film-like quality",
        needs_screenshot=False
    ),
    ScriptSegment(
        id="segment_5",
        start_seconds=52,
        end_seconds=64,
        text="If you need control, use image-to-video: lock a first frame for style and composition, then add motion on top. Great for consistent characters.",
        visual_prompt="{style} shot of the {animal} in a cozy artist workshop, painting on an easel, the painting magically comes to life with gentle movement, warm golden afternoon sunlight through windows, brushes and paint pots around, heartwarming scene",
        needs_screenshot=False
    ),
    ScriptSegment(
        id="segment_6",
        start_seconds=64,
        end_seconds=76,
        text="And if you already have footage, there's reference-to-video: feed short reference clips to keep identity and behavior consistent across generations.",
        visual_prompt="{style} shot of the {animal} in a vintage cinema room with red velvet seats, watching old film reels playing on a classic projector, nostalgic warm lighting, popcorn bucket nearby, cozy movie night atmosphere, charming and whimsical",
        needs_screenshot=False
    ),
    ScriptSegment(
        id="segment_7",
        start_seconds=76,
        end_seconds=88,
        text="Because it's on fal, integration is straightforward: a website button calls the fal API, returns a video URL, and your app can play it instantly — perfect for demos, creatives, and production workflows.",
        visual_prompt="{style} shot of the {animal} at a friendly coffee shop table with a laptop, happily typing and creating, steam rising from a latte, plants and warm string lights in background, productive creative vibes",
        needs_screenshot=False  # No screenshot, just visual
    ),
    ScriptSegment(
        id="segment_8",
        start_seconds=88,
        end_seconds=90,
        text="That's Wan 2.6 on fal: voice-driven characters, multi-scene storytelling, shipped via API.",
        visual_prompt="{style} final shot of the {animal} wearing tropical Hawaiian shirt, giving a thumbs up or wave next to the fal logo, premium outro shot with subtle particles/effects",
        needs_screenshot=False,
        composite_image_urls=["https://images.seeklogo.com/logo-png/61/2/fal-ai-logo-png_seeklogo-611592.png"]
    ),
]

# Default segments (will be replaced by config.get_segments())
SCRIPT_SEGMENTS = SCRIPT_SEGMENTS_TEMPLATE


# Custom segments storage (will be populated dynamically)
_custom_segments: List[ScriptSegment] = []


@dataclass 
class Config:
    """Main configuration for the video generation pipeline."""
    
    # API Configuration
    fal_key: str = field(default_factory=lambda: os.getenv("FAL_KEY", ""))
    voice_id: str = field(default_factory=lambda: os.getenv("VOICE_ID", "Voice4c5cab3d1765912370"))
    
    # Model endpoints (Wan 2.6 + Nano Banana Pro)
    speech_model: str = "fal-ai/minimax/speech-02-hd"
    video_model: str = "wan/v2.6/image-to-video"  # Wan 2.6 image-to-video for character consistency
    text_to_video_model: str = "wan/v2.6/text-to-video"  # Wan 2.6 text-to-video
    image_model: str = "fal-ai/nano-banana-pro"  # Nano Banana Pro for images
    image_edit_model: str = "fal-ai/nano-banana-pro/edit"  # Nano Banana Pro Edit for combining
    
    # Speech settings
    speech_emotion: str = "happy"
    speech_speed: float = 1.0  # 0.5 to 2.0
    speech_max_retries: int = 3  # Max retries for duration adjustment
    
    # Wan 2.6 audio settings
    wan_video_duration: int = 15  # Target video duration in seconds
    
    # Video settings
    video_resolution: str = "720p"  # or "1080p"
    
    # Output directories
    output_dir: str = "output"
    audio_dir: str = "output/audio"
    images_dir: str = "output/images"
    videos_dir: str = "output/videos"
    screenshots_dir: str = "output/screenshots"  # For imported screenshots
    composites_dir: str = "output/composites"    # For combined images (screenshot + character)
    
    # Character settings
    animal_type: str = "capybara"  # Can be changed: capybara, penguin, fox, owl, etc.
    style: str = "3D animation style, cinematic 4K"  # Free text style: "Ghibli", "Moebius", "cartoon", etc.
    custom_character_prompt: str = ""  # Custom prompt override (if empty, uses auto-generated)
    
    def get_character_prompt(self) -> str:
        """Generate character prompt with the selected animal and style."""
        # Use custom prompt if provided
        if self.custom_character_prompt and self.custom_character_prompt.strip():
            return self.custom_character_prompt.strip()
        
        # Auto-generate prompt with animal and style
        return f"""A cute stylized {self.animal_type} character with expressive big eyes, 
    wearing a colorful tropical Hawaiian shirt with palm trees and flowers pattern,
    simple and clean design, {self.style}, consistent character design,
    soft lighting, premium render quality, the {self.animal_type} is anthropomorphized and can gesture/speak,
    friendly chill and professional appearance, relaxed vibe"""
    
    # Legacy property for compatibility
    @property
    def character_prompt(self) -> str:
        return self.get_character_prompt()
    
    # Automation settings
    auto_screenshot: bool = True  # Automatically take screenshots
    reuse_main_character: bool = True  # Reuse main character across all segments
    
    def get_segments(self) -> List[ScriptSegment]:
        """Get script segments with animal type and style substituted."""
        global _custom_segments
        
        # Use custom segments if they exist
        source_segments = _custom_segments if _custom_segments else SCRIPT_SEGMENTS_TEMPLATE
        
        segments = []
        for template in source_segments:
            # Create a new segment with animal and style substituted
            text = template.text.replace("{animal}", self.animal_type).replace("{style}", self.style)
            visual_prompt = template.visual_prompt.replace("{animal}", self.animal_type).replace("{style}", self.style)
            
            segment = ScriptSegment(
                id=template.id,
                start_seconds=template.start_seconds,
                end_seconds=template.end_seconds,
                text=text,
                visual_prompt=visual_prompt,
                screenshot_urls=template.screenshot_urls.copy() if template.screenshot_urls else [],
                needs_screenshot=template.needs_screenshot,
                composite_image_urls=template.composite_image_urls.copy() if hasattr(template, 'composite_image_urls') and template.composite_image_urls else []
            )
            segments.append(segment)
        return segments
    
    def set_custom_segments(self, segments: List[ScriptSegment]):
        """Set custom segments (replacing the default template)."""
        global _custom_segments
        _custom_segments = segments
        self._save_segments_to_file()
    
    def add_segment(self, segment: ScriptSegment):
        """Add a new segment."""
        global _custom_segments
        if not _custom_segments:
            _custom_segments = list(SCRIPT_SEGMENTS_TEMPLATE)
        _custom_segments.append(segment)
        self._save_segments_to_file()
    
    def update_segment(self, segment_id: str, updates: dict) -> bool:
        """Update an existing segment by ID."""
        global _custom_segments
        if not _custom_segments:
            _custom_segments = list(SCRIPT_SEGMENTS_TEMPLATE)
        
        for i, seg in enumerate(_custom_segments):
            if seg.id == segment_id:
                # Update fields
                if 'text' in updates:
                    seg.text = updates['text']
                if 'visual_prompt' in updates:
                    seg.visual_prompt = updates['visual_prompt']
                if 'start_seconds' in updates:
                    seg.start_seconds = updates['start_seconds']
                if 'end_seconds' in updates:
                    seg.end_seconds = updates['end_seconds']
                if 'needs_screenshot' in updates:
                    seg.needs_screenshot = updates['needs_screenshot']
                if 'screenshot_urls' in updates:
                    seg.screenshot_urls = updates['screenshot_urls']
                if 'composite_image_urls' in updates:
                    seg.composite_image_urls = updates['composite_image_urls']
                _custom_segments[i] = seg
                self._save_segments_to_file()
                return True
        return False
    
    def delete_segment(self, segment_id: str) -> bool:
        """Delete a segment by ID."""
        global _custom_segments
        if not _custom_segments:
            _custom_segments = list(SCRIPT_SEGMENTS_TEMPLATE)
        
        for i, seg in enumerate(_custom_segments):
            if seg.id == segment_id:
                _custom_segments.pop(i)
                self._save_segments_to_file()
                return True
        return False
    
    def reset_segments(self):
        """Reset to default template segments."""
        global _custom_segments
        _custom_segments = []
        # Remove saved file
        segments_file = os.path.join(self.output_dir, "custom_segments.json")
        if os.path.exists(segments_file):
            os.remove(segments_file)
    
    def _save_segments_to_file(self):
        """Save custom segments to JSON file."""
        global _custom_segments
        if not _custom_segments:
            return
        
        segments_file = os.path.join(self.output_dir, "custom_segments.json")
        data = []
        for seg in _custom_segments:
            data.append({
                "id": seg.id,
                "start_seconds": seg.start_seconds,
                "end_seconds": seg.end_seconds,
                "text": seg.text,
                "visual_prompt": seg.visual_prompt,
                "screenshot_urls": seg.screenshot_urls,
                "needs_screenshot": seg.needs_screenshot,
                "composite_image_urls": seg.composite_image_urls if hasattr(seg, 'composite_image_urls') else []
            })
        
        with open(segments_file, 'w') as f:
            import json
            json.dump(data, f, indent=2)
    
    def load_segments_from_file(self):
        """Load custom segments from JSON file if it exists."""
        global _custom_segments
        segments_file = os.path.join(self.output_dir, "custom_segments.json")
        
        if os.path.exists(segments_file):
            import json
            with open(segments_file, 'r') as f:
                data = json.load(f)
            
            _custom_segments = []
            for item in data:
                seg = ScriptSegment(
                    id=item['id'],
                    start_seconds=item['start_seconds'],
                    end_seconds=item['end_seconds'],
                    text=item['text'],
                    visual_prompt=item['visual_prompt'],
                    screenshot_urls=item.get('screenshot_urls', []),
                    needs_screenshot=item.get('needs_screenshot', False),
                    composite_image_urls=item.get('composite_image_urls', [])
                )
                _custom_segments.append(seg)
    
    def __post_init__(self):
        """Create output directories if they don't exist."""
        for dir_path in [self.output_dir, self.audio_dir, self.images_dir, 
                         self.videos_dir, self.screenshots_dir, self.composites_dir]:
            os.makedirs(dir_path, exist_ok=True)


def get_config() -> Config:
    """Load configuration from environment."""
    from dotenv import load_dotenv
    load_dotenv()
    return Config()

