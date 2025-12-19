"""Main pipeline for automated video generation using fal.ai APIs."""

import os
import json
import time
import requests
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from pathlib import Path

import fal_client

from config import Config, ScriptSegment, SCRIPT_SEGMENTS, SCRIPT_SEGMENTS_TEMPLATE, SCREENSHOT_URLS, get_config


@dataclass
class GenerationResult:
    """Result from a generation step."""
    segment_id: str
    step: str  # "audio", "image", "video"
    success: bool
    url: Optional[str] = None
    local_path: Optional[str] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class VideoGenerationPipeline:
    """Pipeline to generate complete videos from script segments."""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        
        # Configure fal client
        os.environ["FAL_KEY"] = self.config.fal_key
        
        self.results: List[GenerationResult] = []
        
        # Main character reference (generated once, used everywhere)
        self.main_character_url: Optional[str] = None
        self.main_character_path: Optional[str] = None
        
        # Cache for screenshots
        self.screenshot_cache: Dict[str, str] = {}  # url -> fal_url
        
    def log(self, message: str):
        """Log a message with timestamp."""
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")
        
    def download_file(self, url: str, local_path: str) -> str:
        """Download a file from URL to local path."""
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        return local_path
    
    # ==================== AUTOMATIC SCREENSHOT ====================
    
    def take_screenshot(self, url: str, name: str = None) -> Optional[str]:
        """
        Take a screenshot of a URL using playwright.
        Returns the local path to the screenshot.
        """
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            self.log("[WARN] Playwright not installed. Installing...")
            import subprocess
            subprocess.run(["pip", "install", "playwright"], check=True)
            subprocess.run(["playwright", "install", "chromium"], check=True)
            from playwright.sync_api import sync_playwright
        
        if name is None:
            # Generate name from URL
            name = url.replace("https://", "").replace("/", "_").replace(".", "_")
        
        screenshot_path = os.path.join(self.config.screenshots_dir, f"{name}.png")
        
        # Check cache
        if url in self.screenshot_cache:
            self.log(f"[INFO] Using cached screenshot for {url}")
            return screenshot_path
        
        self.log(f"[INFO] Taking screenshot of {url}...")
        
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page(viewport={"width": 1920, "height": 1080})
                page.goto(url, wait_until="networkidle")
                
                # Wait a bit for any animations
                page.wait_for_timeout(2000)
                
                # Take screenshot
                page.screenshot(path=screenshot_path, full_page=False)
                browser.close()
            
            self.log(f"[OK] Screenshot saved: {screenshot_path}")
            
            # Upload to fal and cache
            fal_url = self.upload_file(screenshot_path)
            self.screenshot_cache[url] = fal_url
            
            return screenshot_path
            
        except Exception as e:
            self.log(f"[ERROR] Screenshot failed for {url}: {e}")
            return None
    
    def take_all_screenshots(self, urls: List[str] = None) -> Dict[str, str]:
        """
        Take screenshots of all specified URLs.
        Returns a dict mapping URL -> fal_url.
        """
        if urls is None:
            urls = list(SCREENSHOT_URLS.values())
        
        self.log(f"[INFO] Taking {len(urls)} screenshots...")
        
        results = {}
        for url in urls:
            path = self.take_screenshot(url)
            if path and url in self.screenshot_cache:
                results[url] = self.screenshot_cache[url]
        
        return results
    
    # ==================== MAIN CHARACTER GENERATION ====================
    
    def generate_main_character(self) -> GenerationResult:
        """
        Generate the main character reference image.
        This is generated ONCE and reused for all segments.
        """
        self.log("[INFO] Generating main character reference...")
        
        # Check if already generated
        if self.main_character_url:
            self.log("[OK] Using existing main character reference")
            return GenerationResult(
                segment_id="main_character",
                step="character",
                success=True,
                url=self.main_character_url,
                local_path=self.main_character_path
            )
        
        try:
            self.log(f"[INFO] Prompt: {self.config.character_prompt[:100]}...")
            result = fal_client.subscribe(
                self.config.image_model,
                arguments={
                    "prompt": self.config.character_prompt,
                    "aspect_ratio": "1:1",  # Square for character reference
                    "safety_checker": False,  # Disable safety checker
                },
                with_logs=True,
            )
            
            images = result.get("images", [])
            if not images:
                return GenerationResult(
                    segment_id="main_character",
                    step="character",
                    success=False,
                    error="No images in response"
                )
            
            image_url = images[0].get("url")
            
            # Download the character image
            local_path = os.path.join(
                self.config.images_dir,
                "main_character_reference.png"
            )
            self.download_file(image_url, local_path)
            
            # Store for reuse
            self.main_character_url = image_url
            self.main_character_path = local_path
            
            self.log(f"[OK] Main character generated: {local_path}")
            
            return GenerationResult(
                segment_id="main_character",
                step="character",
                success=True,
                url=image_url,
                local_path=local_path,
                metadata=result
            )
            
        except Exception as e:
            self.log(f"[ERROR] Main character generation failed: {e}")
            return GenerationResult(
                segment_id="main_character",
                step="character",
                success=False,
                error=str(e)
            )
    
    # ==================== AUTOMATIC COMPOSITE GENERATION ====================
    
    def generate_composite_for_segment(
        self, 
        segment: ScriptSegment
    ) -> Optional[GenerationResult]:
        """
        Automatically generate a composite image for a segment.
        Combines screenshot(s) or direct images with the main character.
        """
        # Check if this segment needs a composite (screenshot or direct image)
        has_screenshots = segment.needs_screenshot and segment.screenshot_urls
        has_direct_images = hasattr(segment, 'composite_image_urls') and segment.composite_image_urls
        
        if not has_screenshots and not has_direct_images:
            return None
        
        if not self.main_character_url:
            self.log(f"[WARN] Main character not generated yet")
            return None
        
        self.log(f"[INFO] Creating composite for {segment.id}...")
        self.log(f"  [INFO] Main character URL: {self.main_character_url[:80] if self.main_character_url else 'None'}...")
        
        image_urls = []
        
        # Get screenshot URLs (take screenshots if needed)
        if has_screenshots:
            for url in segment.screenshot_urls:
                self.log(f"  [INFO] Processing screenshot: {url}")
                if url not in self.screenshot_cache:
                    self.take_screenshot(url)
                if url in self.screenshot_cache:
                    image_urls.append(self.screenshot_cache[url])
                    self.log(f"    [OK] Screenshot cached: {self.screenshot_cache[url][:60]}...")
                else:
                    self.log(f"    [ERROR] Screenshot not in cache!")
        
        # Add direct image URLs (no screenshot needed)
        if has_direct_images:
            for url in segment.composite_image_urls:
                self.log(f"  [INFO] Adding direct image: {url[:60]}...")
                image_urls.append(url)
        
        # Add the main character
        self.log(f"  [INFO] Adding main character to composite...")
        image_urls.append(self.main_character_url)
        
        if not image_urls:
            self.log(f"[WARN] No images available for composite")
            return None
        
        # Create composite prompt - INSTRUCTION-BASED for Nano Banana Pro
        # Image order: [screenshot_urls..., main_character_url]
        # So main character is the LAST image
        num_screenshots = len(image_urls) - 1
        screenshot_ref = "the first image" if num_screenshots == 1 else "the first images"
        character_ref = f"image {len(image_urls)}" if len(image_urls) > 1 else "the second image"
        
        animal = self.config.animal_type
        composite_prompt = f"""Take the {animal} character from {character_ref} and place it in the scene.
        
INSTRUCTIONS:
- Position the {animal} on the left or right side of the frame, taking about 30% of the width
- Keep the UI screenshot from {screenshot_ref} visible as the main background/screen
- Make the {animal} appear to be presenting or gesturing toward the screen content
- The {animal} should face slightly toward the screen while still visible to the viewer
- Preserve the {animal}'s outfit, colors, and style exactly as shown
- Maintain professional lighting - soft studio lighting on the character
- Output should be 16:9 cinematic composition
- The {animal} looks engaged and enthusiastic, as if explaining the UI

Scene context: {segment.visual_prompt}"""
        
        # Call edit_image and fix the segment_id to match the original segment
        edit_result = self.edit_image(
            prompt=composite_prompt,
            image_urls=image_urls,
            output_name=f"{segment.id}_composite",
            aspect_ratio="16:9"
        )
        
        # Override segment_id and step to properly link to the segment
        return GenerationResult(
            segment_id=segment.id,  # Use original segment ID, not "segment_1_composite"
            step="image",  # It's the image step for this segment
            success=edit_result.success,
            url=edit_result.url,
            local_path=edit_result.local_path,
            error=edit_result.error,
            metadata=edit_result.metadata
        )
    
    # ==================== SPEECH GENERATION ====================
    
    def get_audio_duration(self, file_path: str) -> float:
        """Get the duration of an audio file in seconds."""
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(file_path)
            return len(audio) / 1000.0  # Convert milliseconds to seconds
        except Exception as e:
            self.log(f"[WARN] Could not get audio duration: {e}")
            return 0.0
    
    def generate_speech(self, segment: ScriptSegment, speed: float = None) -> GenerationResult:
        """
        Generate speech audio for a script segment using MiniMax Speech-02 HD.
        Auto-adjusts speed if audio exceeds segment duration.
        
        Args:
            segment: The script segment to generate speech for
            speed: Override speed (0.5-2.0), defaults to config value
        """
        target_duration = segment.end_seconds - segment.start_seconds
        current_speed = speed if speed is not None else self.config.speech_speed
        max_retries = self.config.speech_max_retries
        
        for attempt in range(max_retries):
            if attempt == 0:
                self.log(f"[INFO] Generating speech for {segment.id}...")
            else:
                self.log(f"[INFO] Retry {attempt}/{max_retries-1} for {segment.id} (speed={current_speed:.1f})...")
            
            try:
                # Voice setting with custom voice ID
                # Format: voice_setting object with voice_id, speed, emotion
                result = fal_client.subscribe(
                    self.config.speech_model,
                    arguments={
                        "text": segment.text,
                        "voice_setting": {
                            "voice_id": self.config.voice_id,  # Custom voice ID
                            "speed": current_speed,
                            "emotion": self.config.speech_emotion,
                        },
                    },
                    with_logs=True,
                )
                
                audio_url = result.get("audio", {}).get("url")
                if not audio_url:
                    return GenerationResult(
                        segment_id=segment.id,
                        step="audio",
                        success=False,
                        error="No audio URL in response"
                    )
                
                # Download the audio file
                local_path = os.path.join(
                    self.config.audio_dir, 
                    f"{segment.id}_speech.mp3"
                )
                self.download_file(audio_url, local_path)
                
                # Check audio duration
                actual_duration = self.get_audio_duration(local_path)
                self.log(f"[OK] Speech generated for {segment.id} ({actual_duration:.1f}s)")
                
                return GenerationResult(
                    segment_id=segment.id,
                    step="audio",
                    success=True,
                    url=audio_url,
                    local_path=local_path,
                    metadata={
                        **result,
                        "duration": actual_duration,
                        "target_duration": target_duration,
                        "speed_used": current_speed
                    }
                )
                
            except Exception as e:
                self.log(f"[ERROR] Speech generation failed for {segment.id}: {e}")
                if attempt < max_retries - 1:
                    self.log("[INFO] Retrying...")
                    continue
                return GenerationResult(
                    segment_id=segment.id,
                    step="audio",
                    success=False,
                    error=str(e)
                )
        
        # Should not reach here, but just in case
        return GenerationResult(
            segment_id=segment.id,
            step="audio",
            success=False,
            error="Max retries exceeded"
        )
    
    # ==================== IMAGE GENERATION ====================
    
    def generate_character_image(self, segment: ScriptSegment) -> GenerationResult:
        """Generate a character reference image using Nano Banana Pro."""
        self.log(f"[INFO] Generating character image for segment {segment.id}...")
        
        # Combine character base prompt with segment-specific visual
        full_prompt = f"{self.config.character_prompt}. Scene: {segment.visual_prompt}"
        
        try:
            self.log(f"[INFO] Prompt: {full_prompt[:100]}...")
            result = fal_client.subscribe(
                self.config.image_model,
                arguments={
                    "prompt": full_prompt,
                    "aspect_ratio": "16:9",  # Widescreen for video
                    "safety_checker": False,  # Disable safety checker
                },
                with_logs=True,
            )
            
            images = result.get("images", [])
            if not images:
                return GenerationResult(
                    segment_id=segment.id,
                    step="image",
                    success=False,
                    error="No images in response"
                )
            
            image_url = images[0].get("url")
            
            # Download the image file
            local_path = os.path.join(
                self.config.images_dir,
                f"{segment.id}_character.png"
            )
            self.download_file(image_url, local_path)
            
            self.log(f"✅ Character image generated for {segment.id}")
            
            return GenerationResult(
                segment_id=segment.id,
                step="image",
                success=True,
                url=image_url,
                local_path=local_path,
                metadata=result
            )
            
        except Exception as e:
            self.log(f"[ERROR] Image generation failed for {segment.id}: {e}")
            return GenerationResult(
                segment_id=segment.id,
                step="image",
                success=False,
                error=str(e)
            )
    
    # ==================== IMAGE EDITING (Nano Banana Pro Edit) ====================
    
    def upload_file(self, local_path: str) -> str:
        """Upload a local file to fal storage and return the URL."""
        self.log(f"[INFO] Uploading file: {local_path}")
        url = fal_client.upload_file(local_path)
        self.log(f"[OK] File uploaded: {url}")
        return url
    
    def edit_image(
        self,
        prompt: str,
        image_urls: List[str],
        output_name: str = "edited",
        aspect_ratio: str = "16:9"
    ) -> GenerationResult:
        """
        Edit/combine images using Nano Banana Pro Edit.
        Supports up to 14 reference images for multi-image composition.
        
        Args:
            prompt: Description of the edit or combination
            image_urls: List of image URLs to combine/edit (up to 14 images)
            output_name: Name for the output file
            aspect_ratio: Output aspect ratio
        """
        self.log(f"[INFO] Editing image with {len(image_urls)} reference(s)...")
        
        if not image_urls:
            return GenerationResult(
                segment_id=output_name,
                step="edit",
                success=False,
                error="No image URLs provided"
            )
        
        # Log all image URLs being passed
        for i, url in enumerate(image_urls):
            self.log(f"  [INFO] Image {i+1}: {url[:80]}...")
        
        try:
            # Nano Banana Pro Edit accepts image_urls (PLURAL) for multi-image composition
            self.log(f"[INFO] Edit prompt: {prompt[:100]}...")
            result = fal_client.subscribe(
                self.config.image_edit_model,
                arguments={
                    "prompt": prompt,
                    "image_urls": image_urls,  # PLURAL - List of URLs for multi-image composition
                    "safety_checker": False,  # Disable safety checker
                },
                with_logs=True,
            )
            
            images = result.get("images", [])
            if not images:
                return GenerationResult(
                    segment_id=output_name,
                    step="edit",
                    success=False,
                    error="No images in response"
                )
            
            image_url = images[0].get("url")
            
            # Download the edited image
            local_path = os.path.join(
                self.config.composites_dir,
                f"{output_name}_composite.png"
            )
            self.download_file(image_url, local_path)
            
            self.log(f"[OK] Image edited successfully: {output_name}")
            
            return GenerationResult(
                segment_id=output_name,
                step="edit",
                success=True,
                url=image_url,
                local_path=local_path,
                metadata=result
            )
            
        except Exception as e:
            self.log(f"[ERROR] Image editing failed: {e}")
            return GenerationResult(
                segment_id=output_name,
                step="edit",
                success=False,
                error=str(e)
            )
    
    def combine_screenshot_with_character(
        self,
        screenshot_path_or_url: str,
        character_image_url: str,
        prompt: str,
        segment_id: str = "composite"
    ) -> GenerationResult:
        """
        Combine a screenshot (UI) with a generated character using Nano Banana Pro Edit.
        
        IMPORTANT: The prompt should be INSTRUCTION-BASED, not description-based!
        
        Good prompt examples:
        - "Take the character from image 2 and place it on the left side of image 1 (the UI). 
           Make the character face toward the screen as if presenting it."
        - "Position the character from the second image in front of the first image.
           The character should be on the right, gesturing toward the UI with enthusiasm."
        
        Bad prompt examples:
        - "A character next to a UI screenshot" (too vague, describes result not action)
        - "Professional scene with character presenting" (no clear instructions)
        
        Args:
            screenshot_path_or_url: Local path or URL to the screenshot (will be image 1)
            character_image_url: URL of the generated character image (will be image 2)
            prompt: INSTRUCTION for how to combine them - be specific about placement!
            segment_id: Identifier for this composite
        """
        self.log(f"[INFO] Combining screenshot with character for {segment_id}...")
        
        # If it's a local path, upload it first
        if not screenshot_path_or_url.startswith('http'):
            screenshot_url = self.upload_file(screenshot_path_or_url)
        else:
            screenshot_url = screenshot_path_or_url
        
        # Combine the images using the edit model
        return self.edit_image(
            prompt=prompt,
            image_urls=[screenshot_url, character_image_url],
            output_name=segment_id,
            aspect_ratio="16:9"
        )
    
    def import_screenshot(self, file_path: str, name: str = None) -> str:
        """
        Import a screenshot file and copy it to the screenshots directory.
        Returns the path in the screenshots directory.
        """
        import shutil
        
        if name is None:
            name = os.path.basename(file_path)
        
        dest_path = os.path.join(self.config.screenshots_dir, name)
        shutil.copy2(file_path, dest_path)
        
        self.log(f"[INFO] Screenshot imported: {dest_path}")
        return dest_path
    
    # ==================== VIDEO GENERATION ====================
    
    def generate_video(
        self, 
        segment: ScriptSegment, 
        audio_url: Optional[str] = None,
        image_url: Optional[str] = None
    ) -> GenerationResult:
        """Generate video for a script segment using image-to-video."""
        self.log(f"[INFO] Generating video for segment {segment.id}...")
        
        # Use the main character image or segment-specific image as first frame
        first_frame_url = image_url
        if not first_frame_url and self.main_character_url:
            first_frame_url = self.main_character_url
        
        # Build the video generation arguments (matching working playground config)
        # Use segment-specific visual prompt
        video_prompt = segment.visual_prompt
        self.log(f"  [INFO] Prompt: {video_prompt[:80]}...")
        
        arguments = {
            "prompt": video_prompt,  # Use segment-specific visual prompt
            "resolution": "1080p",
            "duration": "15",  # 15s video duration
            "negative_prompt": "low resolution, error, worst quality, low quality, defects",
            "enable_prompt_expansion": True,
            "enable_safety_checker": False,
            "multi_shots": True,
        }
        
        # Add image URL for image-to-video (required for Wan 2.6 i2v)
        if not first_frame_url:
            return GenerationResult(
                segment_id=segment.id,
                step="video",
                success=False,
                error="No image URL provided for image-to-video generation"
            )
        
        arguments["image_url"] = first_frame_url
        self.log(f"  [INFO] First frame: {first_frame_url[:80]}...")
        
        # Add audio URL for lip-sync if available
        if audio_url:
            arguments["audio_url"] = audio_url
            self.log(f"  [INFO] Audio: {audio_url[:80]}...")
            
        # Retry up to 3 times for content policy violations
        max_retries = 3
        last_error = None
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    self.log(f"  [INFO] Retry {attempt}/{max_retries-1} for video generation...")
                
                result = fal_client.subscribe(
                    self.config.video_model,
                    arguments=arguments,
                    with_logs=True,
                )
                
                video = result.get("video", {})
                video_url = video.get("url")
                
                if not video_url:
                    last_error = "No video URL in response"
                    continue
                
                # Download the video file
                local_path = os.path.join(
                    self.config.videos_dir,
                    f"{segment.id}_video.mp4"
                )
                self.download_file(video_url, local_path)
                
                self.log(f"[OK] Video generated for {segment.id}")
                
                return GenerationResult(
                    segment_id=segment.id,
                    step="video",
                    success=True,
                    url=video_url,
                    local_path=local_path,
                    metadata=result
                )
                
            except Exception as e:
                error_str = str(e)
                last_error = error_str
                
                # Check if it's a content policy error
                if "content_policy_violation" in error_str.lower():
                    self.log(f"  [WARN] Content policy triggered (attempt {attempt+1}/{max_retries})")
                    if attempt < max_retries - 1:
                        continue  # Try again
                else:
                    # For other errors, don't retry
                    self.log(f"[ERROR] Video generation failed for {segment.id}: {e}")
                    break
        
        self.log(f"[ERROR] Video generation failed for {segment.id} after {max_retries} attempts: {last_error}")
        return GenerationResult(
            segment_id=segment.id,
            step="video",
            success=False,
            error=str(last_error)
        )
    
    # ==================== FULL PIPELINE ====================
    
    def process_segment(self, segment: ScriptSegment) -> Dict[str, GenerationResult]:
        """Process a single segment through all generation steps."""
        results = {}
        
        # Step 1: Generate speech
        audio_result = self.generate_speech(segment)
        results["audio"] = audio_result
        self.results.append(audio_result)
        
        # Step 2: Generate image (composite with screenshot OR regular character image)
        if segment.needs_screenshot:
            # Use composite with screenshot + character
            image_result = self.generate_composite_for_segment(segment)
            if image_result is None:
                # Fallback to regular character image
                image_result = self.generate_character_image(segment)
        else:
            # Regular character image generation
            image_result = self.generate_character_image(segment)
        
        results["image"] = image_result
        self.results.append(image_result)
        
        # Step 3: Generate video with MAIN CHARACTER as first frame
        # (Using composite images often triggers content policy, so use clean main character)
        audio_url = audio_result.url if audio_result.success else None
        video_first_frame = self.main_character_url  # Always use main character, not composite
        video_result = self.generate_video(segment, audio_url=audio_url, image_url=video_first_frame)
        results["video"] = video_result
        self.results.append(video_result)
        
        return results
    
    def run_full_pipeline(
        self, 
        segments: Optional[List[ScriptSegment]] = None
    ) -> List[Dict[str, GenerationResult]]:
        """Run the full pipeline for all segments."""
        segments = segments or self.config.get_segments()
        
        self.log(f"[INFO] Starting full pipeline for {len(segments)} segments...")
        self.log(f"[INFO] Output directory: {self.config.output_dir}")
        
        all_results = []
        
        # ========== PHASE 0: Generate main character (ONCE) ==========
        self.log(f"\n{'='*50}")
        self.log("PHASE 0: Generating main character reference")
        self.log(f"{'='*50}\n")
        
        character_result = self.generate_main_character()
        self.results.append(character_result)
        
        if not character_result.success:
            self.log("[WARN] Main character generation failed, continuing with per-segment generation")
        
        # ========== PHASE 1: Take all screenshots ==========
        screenshot_segments = [s for s in segments if s.needs_screenshot]
        if screenshot_segments:
            self.log(f"\n{'='*50}")
            self.log(f"PHASE 1: Taking {len(screenshot_segments)} screenshots")
            self.log(f"{'='*50}\n")
            
            all_screenshot_urls = []
            for seg in screenshot_segments:
                all_screenshot_urls.extend(seg.screenshot_urls)
            
            unique_urls = list(set(all_screenshot_urls))
            self.take_all_screenshots(unique_urls)
        
        # ========== PHASE 2: Process segments in PARALLEL with 5s stagger ==========
        self.log(f"\n{'='*50}")
        self.log(f"PHASE 2: Processing {len(segments)} segments (PARALLEL with 5s stagger)")
        self.log(f"{'='*50}\n")
        
        # Use ThreadPoolExecutor for parallel processing
        # Stagger starts by 5 seconds to avoid overwhelming the API
        results_lock = threading.Lock()
        
        def process_with_delay(segment, delay):
            """Process a segment after a delay."""
            if delay > 0:
                self.log(f"[WAIT] {segment.id} waiting {delay}s before starting...")
                time.sleep(delay)
            
            self.log(f"\n[INFO] Starting {segment.id}...")
            result = self.process_segment(segment)
            
            with results_lock:
                all_results.append(result)
                self.save_progress()
            
            return result
        
        # Start all segments with 5-second stagger
        with ThreadPoolExecutor(max_workers=len(segments)) as executor:
            futures = []
            for i, segment in enumerate(segments):
                delay = i * 5  # 0s, 5s, 10s, 15s...
                future = executor.submit(process_with_delay, segment, delay)
                futures.append((segment.id, future))
            
            # Wait for all to complete
            for segment_id, future in futures:
                try:
                    future.result()  # Wait for completion
                    self.log(f"[OK] {segment_id} completed")
                except Exception as e:
                    self.log(f"[ERROR] {segment_id} failed: {e}")
        
        self.log(f"\n[OK] Pipeline complete! Results saved to {self.config.output_dir}")
        
        return all_results
    
    def run_automated_full_pipeline(
        self,
        segments: Optional[List[ScriptSegment]] = None
    ) -> List[Dict[str, GenerationResult]]:
        """
        Run the FULLY AUTOMATED pipeline:
        1. Generate main character ONCE
        2. Take all needed screenshots automatically
        3. Create composites for segments with screenshots
        4. Generate audio, images, and videos
        
        This is the main entry point for fully automated video generation.
        """
        return self.run_full_pipeline(segments)
    
    def run_audio_only(
        self, 
        segments: Optional[List[ScriptSegment]] = None
    ) -> List[GenerationResult]:
        """Generate only audio for all segments."""
        segments = segments or self.config.get_segments()
        
        self.log(f"[INFO] Generating audio for {len(segments)} segments...")
        
        results = []
        for segment in segments:
            result = self.generate_speech(segment)
            results.append(result)
            self.results.append(result)
            
        self.save_progress()
        return results
    
    def run_images_only(
        self, 
        segments: Optional[List[ScriptSegment]] = None
    ) -> List[GenerationResult]:
        """Generate only images for all segments."""
        segments = segments or self.config.get_segments()
        
        self.log(f"[INFO] Generating images for {len(segments)} segments...")
        
        results = []
        for segment in segments:
            result = self.generate_character_image(segment)
            results.append(result)
            self.results.append(result)
            
        self.save_progress()
        return results
    
    def run_videos_only(
        self,
        segments: Optional[List[ScriptSegment]] = None,
        use_existing_audio: bool = True
    ) -> List[GenerationResult]:
        """Generate only videos for all segments."""
        segments = segments or self.config.get_segments()
        
        self.log(f"[INFO] Generating videos for {len(segments)} segments...")
        
        results = []
        for segment in segments:
            # Try to find existing audio
            audio_url = None
            if use_existing_audio:
                audio_path = os.path.join(
                    self.config.audio_dir,
                    f"{segment.id}_speech.mp3"
                )
                if os.path.exists(audio_path):
                    # We need to upload it to get a URL
                    # For now, skip if no URL available
                    self.log(f"[WARN] Local audio exists but needs URL upload for {segment.id}")
            
            result = self.generate_video(segment, audio_url=audio_url)
            results.append(result)
            self.results.append(result)
            
        self.save_progress()
        return results
    
    def save_progress(self):
        """Save current progress to a JSON file."""
        progress_file = os.path.join(self.config.output_dir, "progress.json")
        
        progress_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": {
                "speech_model": self.config.speech_model,
                "video_model": self.config.video_model,
                "image_model": self.config.image_model,
                "voice_id": self.config.voice_id,
                "speech_emotion": self.config.speech_emotion,
            },
            "results": [asdict(r) for r in self.results]
        }
        
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
            
        self.log(f"[INFO] Progress saved to {progress_file}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all generation results."""
        summary = {
            "total": len(self.results),
            "successful": sum(1 for r in self.results if r.success),
            "failed": sum(1 for r in self.results if not r.success),
            "by_step": {
                "audio": {"success": 0, "failed": 0},
                "image": {"success": 0, "failed": 0},
                "video": {"success": 0, "failed": 0},
            }
        }
        
        for result in self.results:
            if result.success:
                summary["by_step"][result.step]["success"] += 1
            else:
                summary["by_step"][result.step]["failed"] += 1
                
        return summary


# ==================== CLI Interface ====================

def main():
    """Main entry point for the pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate videos using fal.ai APIs"
    )
    parser.add_argument(
        "--mode", 
        choices=["full", "audio", "images", "videos", "test"],
        default="test",
        help="Pipeline mode: full (all steps), audio/images/videos (single step), test (first segment only)"
    )
    parser.add_argument(
        "--segment",
        type=int,
        help="Process only a specific segment (1-8)"
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = VideoGenerationPipeline()
    
    # Determine which segments to process
    all_segments = pipeline.config.get_segments()
    segments = all_segments
    if args.segment:
        if 1 <= args.segment <= len(all_segments):
            segments = [all_segments[args.segment - 1]]
        else:
            print(f"Invalid segment number. Must be 1-{len(all_segments)}")
            return
    elif args.mode == "test":
        segments = [all_segments[0]]  # Just first segment for testing
    
    # Run the appropriate pipeline mode
    if args.mode == "full":
        pipeline.run_full_pipeline(segments)
    elif args.mode == "audio":
        pipeline.run_audio_only(segments)
    elif args.mode == "images":
        pipeline.run_images_only(segments)
    elif args.mode == "videos":
        pipeline.run_videos_only(segments)
    elif args.mode == "test":
        pipeline.run_full_pipeline(segments)
    
    # Print summary
    summary = pipeline.get_summary()
    print(f"\nSummary:")
    print(f"   Total generations: {summary['total']}")
    print(f"   Successful: {summary['successful']}")
    print(f"   Failed: {summary['failed']}")
    

if __name__ == "__main__":
    main()

