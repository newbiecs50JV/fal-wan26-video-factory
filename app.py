"""Flask web application for the video generation pipeline."""

import os
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, render_template, jsonify, request, send_from_directory
from dotenv import load_dotenv

from config import Config, SCRIPT_SEGMENTS, SCRIPT_SEGMENTS_TEMPLATE, get_config
from pipeline import VideoGenerationPipeline, GenerationResult

# Load environment variables
load_dotenv()

app = Flask(__name__, static_folder='static', template_folder='templates')

# Global pipeline instance and state
pipeline = None
generation_status = {
    "running": False,
    "cancelled": False,
    "current_segment": None,
    "current_step": None,
    "progress": 0,
    "results": [],
    "errors": [],
    "logs": []
}

# Dynamic API configuration (can be set via UI)
api_config = {
    "fal_key": os.getenv("FAL_KEY", ""),
    "voice_id": os.getenv("VOICE_ID", "Voice4c5cab3d1765912370")
}

# Lock for thread-safe log updates
import threading as _threading
log_lock = _threading.Lock()

def add_log(message: str):
    """Add a log message to the status."""
    import time
    timestamp = time.strftime("%H:%M:%S")
    with log_lock:
        generation_status["logs"].append(f"[{timestamp}] {message}")
        # Keep only last 100 logs
        if len(generation_status["logs"]) > 100:
            generation_status["logs"] = generation_status["logs"][-100:]


def get_pipeline():
    """Get or create the pipeline instance with current API config."""
    global pipeline
    if pipeline is None:
        # Create pipeline with current API config
        config = get_config()
        if api_config["fal_key"]:
            config.fal_key = api_config["fal_key"]
        if api_config["voice_id"]:
            config.voice_id = api_config["voice_id"]
        # Load any saved custom segments
        config.load_segments_from_file()
        pipeline = VideoGenerationPipeline(config)
    return pipeline


def reset_pipeline():
    """Reset the pipeline to use new config."""
    global pipeline
    pipeline = None


@app.route('/')
def index():
    """Main page with the video generation interface."""
    p = get_pipeline()
    return render_template('index.html', segments=p.config.get_segments())


@app.route('/api/segments')
def get_segments():
    """Get all script segments."""
    p = get_pipeline()
    segments_data = []
    for seg in p.config.get_segments():
        segments_data.append({
            "id": seg.id,
            "start": seg.start_seconds,
            "end": seg.end_seconds,
            "duration": seg.end_seconds - seg.start_seconds,
            "text": seg.text,
            "visual_prompt": seg.visual_prompt,
            "needs_screenshot": seg.needs_screenshot,
            "screenshot_urls": seg.screenshot_urls if seg.needs_screenshot else [],
            "composite_image_urls": seg.composite_image_urls if hasattr(seg, 'composite_image_urls') else []
        })
    return jsonify(segments_data)


@app.route('/api/segments', methods=['POST'])
def add_segment():
    """Add a new segment."""
    from config import ScriptSegment
    
    data = request.get_json() or {}
    
    # Validate required fields
    if not data.get('id'):
        return jsonify({"error": "id is required"}), 400
    if not data.get('text'):
        return jsonify({"error": "text is required"}), 400
    
    # Create new segment
    segment = ScriptSegment(
        id=data['id'],
        start_seconds=data.get('start', 0),
        end_seconds=data.get('end', 10),
        text=data['text'],
        visual_prompt=data.get('visual_prompt', ''),
        screenshot_urls=data.get('screenshot_urls', []),
        needs_screenshot=data.get('needs_screenshot', False),
        composite_image_urls=data.get('composite_image_urls', [])
    )
    
    p = get_pipeline()
    p.config.add_segment(segment)
    
    add_log(f"[INFO] Segment added: {segment.id}")
    return jsonify({"message": f"Segment {segment.id} added", "segment": data})


@app.route('/api/segments/<segment_id>', methods=['PUT'])
def update_segment(segment_id):
    """Update an existing segment."""
    data = request.get_json() or {}
    
    p = get_pipeline()
    success = p.config.update_segment(segment_id, data)
    
    if success:
        add_log(f"[INFO] Segment updated: {segment_id}")
        return jsonify({"message": f"Segment {segment_id} updated"})
    else:
        return jsonify({"error": f"Segment {segment_id} not found"}), 404


@app.route('/api/segments/<segment_id>', methods=['DELETE'])
def delete_segment(segment_id):
    """Delete a segment."""
    p = get_pipeline()
    success = p.config.delete_segment(segment_id)
    
    if success:
        add_log(f"[INFO] Segment deleted: {segment_id}")
        return jsonify({"message": f"Segment {segment_id} deleted"})
    else:
        return jsonify({"error": f"Segment {segment_id} not found"}), 404


@app.route('/api/segments/reset', methods=['POST'])
def reset_segments():
    """Reset segments to default template."""
    p = get_pipeline()
    p.config.reset_segments()
    
    add_log("[INFO] Segments reset to default template")
    return jsonify({"message": "Segments reset to default"})


@app.route('/api/status')
def get_status():
    """Get current generation status."""
    p = get_pipeline()
    return jsonify({
        **generation_status,
        "animal_type": p.config.animal_type,
        "style": p.config.style
    })


@app.route('/api/animal', methods=['GET', 'POST'])
def animal_type():
    """Get or set the animal type for the character."""
    p = get_pipeline()
    
    if request.method == 'POST':
        data = request.get_json() or {}
        new_animal = data.get('animal_type', '').strip()
        if new_animal:
            p.config.animal_type = new_animal
            # Clear main character so it regenerates with new animal
            p.main_character_url = None
            p.main_character_path = None
            add_log(f"[INFO] Animal changed to: {new_animal}")
            return jsonify({"message": f"Animal changed to {new_animal}", "animal_type": new_animal})
        return jsonify({"error": "animal_type required"}), 400
    
    return jsonify({"animal_type": p.config.animal_type})


@app.route('/api/style', methods=['GET', 'POST'])
def style_type():
    """Get or set the style for the character and scenes."""
    p = get_pipeline()
    
    if request.method == 'POST':
        data = request.get_json() or {}
        new_style = data.get('style', '').strip()
        if new_style:
            p.config.style = new_style
            # Clear main character so it regenerates with new style
            p.main_character_url = None
            p.main_character_path = None
            add_log(f"[INFO] Style changed to: {new_style}")
            return jsonify({"message": f"Style changed to {new_style}", "style": new_style})
        return jsonify({"error": "style required"}), 400
    
    return jsonify({"style": p.config.style})


# ==================== API KEY CONFIGURATION ====================

@app.route('/api/config/keys', methods=['GET', 'POST'])
def config_keys():
    """Get or set API keys (FAL_KEY, VOICE_ID)."""
    global api_config
    
    if request.method == 'POST':
        data = request.get_json() or {}
        
        # Update FAL_KEY if provided
        if 'fal_key' in data:
            api_config["fal_key"] = data['fal_key'].strip()
            os.environ["FAL_KEY"] = api_config["fal_key"]
        
        # Update VOICE_ID if provided
        if 'voice_id' in data:
            api_config["voice_id"] = data['voice_id'].strip()
        
        # Reset pipeline to use new config
        reset_pipeline()
        
        add_log("[INFO] API configuration updated")
        return jsonify({
            "message": "API configuration updated",
            "has_fal_key": bool(api_config["fal_key"]),
            "voice_id": api_config["voice_id"]
        })
    
    # GET - return current config (mask the key)
    return jsonify({
        "has_fal_key": bool(api_config["fal_key"]),
        "fal_key_preview": api_config["fal_key"][:8] + "..." if len(api_config["fal_key"]) > 8 else "",
        "voice_id": api_config["voice_id"]
    })


@app.route('/api/config/keys/status')
def config_keys_status():
    """Check if API keys are configured and valid."""
    has_key = bool(api_config["fal_key"])
    
    # Try a simple validation by checking if pipeline can be created
    is_valid = False
    if has_key:
        try:
            p = get_pipeline()
            is_valid = True
        except Exception as e:
            is_valid = False
    
    return jsonify({
        "configured": has_key,
        "valid": is_valid,
        "voice_id": api_config["voice_id"]
    })


@app.route('/api/character-prompt', methods=['GET', 'POST'])
def character_prompt():
    """Get or set the custom character prompt."""
    p = get_pipeline()
    
    if request.method == 'POST':
        data = request.get_json() or {}
        custom_prompt = data.get('prompt', '')
        p.config.custom_character_prompt = custom_prompt
        # Clear main character so it regenerates with new prompt
        p.main_character_url = None
        p.main_character_path = None
        if custom_prompt:
            add_log(f"[INFO] Custom character prompt set")
        else:
            add_log(f"[INFO] Using auto-generated character prompt")
        return jsonify({
            "message": "Character prompt updated",
            "custom_prompt": custom_prompt,
            "effective_prompt": p.config.get_character_prompt()
        })
    
    return jsonify({
        "custom_prompt": p.config.custom_character_prompt,
        "effective_prompt": p.config.get_character_prompt()
    })


@app.route('/api/cancel', methods=['POST'])
def cancel_generation():
    """Cancel the current generation process."""
    global generation_status
    if generation_status["running"]:
        generation_status["cancelled"] = True
        add_log("[WARN] Cancellation requested...")
        return jsonify({"message": "Cancellation requested"})
    return jsonify({"error": "No process running"}), 400


@app.route('/api/logs')
def get_logs():
    """Get generation logs."""
    return jsonify({"logs": generation_status["logs"]})


@app.route('/api/logs/clear', methods=['POST'])
def clear_logs():
    """Clear all logs."""
    global generation_status
    generation_status["logs"] = []
    return jsonify({"message": "Logs cleared"})


@app.route('/api/generate/character', methods=['POST'])
def generate_character():
    """Generate and preview the main character before running full pipeline."""
    if generation_status["running"]:
        return jsonify({"error": "Generation already in progress"}), 400
    
    def run_character():
        global generation_status
        generation_status["running"] = True
        generation_status["results"] = []
        generation_status["errors"] = []
        generation_status["current_segment"] = "main_character"
        generation_status["current_step"] = "character"
        generation_status["progress"] = 0
        
        try:
            p = get_pipeline()
            
            result = p.generate_main_character()
            generation_status["results"].append({
                "segment_id": result.segment_id,
                "step": result.step,
                "success": result.success,
                "url": result.url,
                "local_path": result.local_path,
                "error": result.error
            })
            
            if not result.success:
                generation_status["errors"].append(f"Main character: {result.error}")
            
            generation_status["progress"] = 100
            
        finally:
            generation_status["running"] = False
            generation_status["current_segment"] = None
            generation_status["current_step"] = None
    
    thread = threading.Thread(target=run_character)
    thread.start()
    
    return jsonify({"message": "Generating main character for preview..."})


@app.route('/api/character/status')
def character_status():
    """Check if main character has been generated and approved."""
    p = get_pipeline()
    return jsonify({
        "generated": p.main_character_url is not None,
        "url": p.main_character_url,
        "local_path": p.main_character_path
    })


@app.route('/api/character/regenerate', methods=['POST'])
def regenerate_character():
    """Regenerate the main character with a new prompt."""
    data = request.json or {}
    new_prompt = data.get('prompt')
    
    if generation_status["running"]:
        return jsonify({"error": "Generation already in progress"}), 400
    
    p = get_pipeline()
    
    # Reset the character
    p.main_character_url = None
    p.main_character_path = None
    
    # Update prompt if provided
    if new_prompt:
        p.config.character_prompt = new_prompt
    
    # Generate new character
    return generate_character()


@app.route('/api/generate/audio', methods=['POST'])
def generate_audio():
    """Generate audio for specified segments."""
    data = request.json or {}
    segment_ids = data.get('segment_ids', [])
    
    if generation_status["running"]:
        return jsonify({"error": "Generation already in progress"}), 400
    
    # Filter segments
    p = get_pipeline()
    all_segments = p.config.get_segments()
    segments = all_segments
    if segment_ids:
        segments = [s for s in all_segments if s.id in segment_ids]
    
    # Run in background thread
    def run_audio():
        global generation_status
        generation_status["running"] = True
        generation_status["results"] = []
        generation_status["errors"] = []
        
        try:
            p = get_pipeline()
            for i, segment in enumerate(segments):
                generation_status["current_segment"] = segment.id
                generation_status["current_step"] = "audio"
                generation_status["progress"] = int((i / len(segments)) * 100)
                
                result = p.generate_speech(segment)
                generation_status["results"].append({
                    "segment_id": result.segment_id,
                    "step": result.step,
                    "success": result.success,
                    "url": result.url,
                    "local_path": result.local_path,
                    "error": result.error
                })
                
                if not result.success:
                    generation_status["errors"].append(f"{segment.id}: {result.error}")
                    
            generation_status["progress"] = 100
        finally:
            generation_status["running"] = False
            generation_status["current_segment"] = None
            generation_status["current_step"] = None
    
    thread = threading.Thread(target=run_audio)
    thread.start()
    
    return jsonify({"message": "Audio generation started", "segments": len(segments)})


@app.route('/api/generate/images', methods=['POST'])
def generate_images():
    """Generate character images for specified segments."""
    data = request.json or {}
    segment_ids = data.get('segment_ids', [])
    
    if generation_status["running"]:
        return jsonify({"error": "Generation already in progress"}), 400
    
    p = get_pipeline()
    all_segments = p.config.get_segments()
    segments = all_segments
    if segment_ids:
        segments = [s for s in all_segments if s.id in segment_ids]
    
    def run_images():
        global generation_status
        generation_status["running"] = True
        generation_status["results"] = []
        generation_status["errors"] = []
        
        try:
            p = get_pipeline()
            for i, segment in enumerate(segments):
                generation_status["current_segment"] = segment.id
                generation_status["current_step"] = "image"
                generation_status["progress"] = int((i / len(segments)) * 100)
                
                result = p.generate_character_image(segment)
                generation_status["results"].append({
                    "segment_id": result.segment_id,
                    "step": result.step,
                    "success": result.success,
                    "url": result.url,
                    "local_path": result.local_path,
                    "error": result.error
                })
                
                if not result.success:
                    generation_status["errors"].append(f"{segment.id}: {result.error}")
                    
            generation_status["progress"] = 100
        finally:
            generation_status["running"] = False
            generation_status["current_segment"] = None
            generation_status["current_step"] = None
    
    thread = threading.Thread(target=run_images)
    thread.start()
    
    return jsonify({"message": "Image generation started", "segments": len(segments)})


@app.route('/api/generate/videos', methods=['POST'])
def generate_videos():
    """Generate videos for specified segments."""
    data = request.json or {}
    segment_ids = data.get('segment_ids', [])
    
    if generation_status["running"]:
        return jsonify({"error": "Generation already in progress"}), 400
    
    p = get_pipeline()
    all_segments = p.config.get_segments()
    segments = all_segments
    if segment_ids:
        segments = [s for s in all_segments if s.id in segment_ids]
    
    def run_videos():
        global generation_status
        generation_status["running"] = True
        generation_status["results"] = []
        generation_status["errors"] = []
        
        try:
            p = get_pipeline()
            
            # First, check for existing audio files and get their URLs
            audio_results = {}
            for result in p.results:
                if result.step == "audio" and result.success and result.url:
                    audio_results[result.segment_id] = result.url
            
            for i, segment in enumerate(segments):
                generation_status["current_segment"] = segment.id
                generation_status["current_step"] = "video"
                generation_status["progress"] = int((i / len(segments)) * 100)
                
                audio_url = audio_results.get(segment.id)
                result = p.generate_video(segment, audio_url=audio_url)
                generation_status["results"].append({
                    "segment_id": result.segment_id,
                    "step": result.step,
                    "success": result.success,
                    "url": result.url,
                    "local_path": result.local_path,
                    "error": result.error
                })
                
                if not result.success:
                    generation_status["errors"].append(f"{segment.id}: {result.error}")
                    
            generation_status["progress"] = 100
        finally:
            generation_status["running"] = False
            generation_status["current_segment"] = None
            generation_status["current_step"] = None
    
    thread = threading.Thread(target=run_videos)
    thread.start()
    
    return jsonify({"message": "Video generation started", "segments": len(segments)})


@app.route('/api/generate/full', methods=['POST'])
def generate_full():
    """Run the FULLY AUTOMATED pipeline for specified segments."""
    data = request.json or {}
    segment_ids = data.get('segment_ids', [])
    
    if generation_status["running"]:
        return jsonify({"error": "Generation already in progress"}), 400
    
    p = get_pipeline()
    all_segments = p.config.get_segments()
    segments = all_segments
    if segment_ids:
        segments = [s for s in all_segments if s.id in segment_ids]
    
    # Count total steps: character + screenshots + (audio + image + video) per segment
    screenshot_segments = [s for s in segments if s.needs_screenshot]
    total_steps = 1 + len(set(url for s in screenshot_segments for url in s.screenshot_urls)) + len(segments) * 3
    
    def run_full():
        global generation_status
        generation_status["running"] = True
        generation_status["cancelled"] = False
        generation_status["results"] = []
        generation_status["errors"] = []
        generation_status["logs"] = []
        current_step = 0
        
        try:
            p = get_pipeline()
            add_log("[INFO] Starting fully automated pipeline...")
            
            # ========== PHASE 0: Check main character exists ==========
            if not p.main_character_url:
                # Generate if not already validated
                add_log("[INFO] Generating main character...")
                generation_status["current_segment"] = "setup"
                generation_status["current_step"] = "character"
                generation_status["progress"] = int((current_step / total_steps) * 100)
                
                character_result = p.generate_main_character()
                generation_status["results"].append({
                    "segment_id": character_result.segment_id,
                    "step": character_result.step,
                    "success": character_result.success,
                    "url": character_result.url,
                    "local_path": character_result.local_path,
                    "error": character_result.error
                })
                add_log(f"{'[OK]' if character_result.success else '[ERROR]'} Main character: {'ready' if character_result.success else character_result.error}")
                
                if not character_result.success:
                    generation_status["errors"].append(f"main_character: {character_result.error}")
            else:
                # Character already validated, skip generation
                add_log("[OK] Using pre-validated main character")
                generation_status["results"].append({
                    "segment_id": "main_character",
                    "step": "character",
                    "success": True,
                    "url": p.main_character_url,
                    "local_path": p.main_character_path,
                    "error": None
                })
            current_step += 1
            
            if generation_status["cancelled"]:
                add_log("[WARN] Process cancelled by user")
                return
            
            # ========== PHASE 1: Take all screenshots automatically ==========
            if screenshot_segments:
                add_log(f"[INFO] Taking {len(screenshot_segments)} screenshots...")
                generation_status["current_step"] = "screenshots"
                
                all_urls = list(set(url for s in screenshot_segments for url in s.screenshot_urls))
                for url in all_urls:
                    if generation_status["cancelled"]:
                        add_log("[WARN] Process cancelled by user")
                        return
                    
                    add_log(f"[INFO] Screenshot: {url[:50]}...")
                    generation_status["current_segment"] = f"screenshot"
                    generation_status["progress"] = int((current_step / total_steps) * 100)
                    
                    try:
                        p.take_screenshot(url)
                        add_log(f"[OK] Screenshot captured")
                    except Exception as e:
                        add_log(f"[ERROR] Screenshot failed: {e}")
                        generation_status["errors"].append(f"screenshot: {e}")
                    
                    current_step += 1
            
            # ========== PHASE 2: Process segments in PARALLEL with 5s stagger ==========
            add_log(f"[INFO] Starting PARALLEL processing of {len(segments)} segments (5s stagger)")
            
            results_lock = threading.Lock()
            completed_count = [0]  # Use list to allow modification in nested function
            
            def process_segment_parallel(segment, delay):
                """Process a single segment after a delay."""
                nonlocal current_step
                
                if delay > 0:
                    add_log(f"[WAIT] {segment.id} waiting {delay}s...")
                    time.sleep(delay)
                
                if generation_status["cancelled"]:
                    return
                
                add_log(f"[INFO] Starting {segment.id}...")
                
                # Audio
                add_log(f"[INFO] [{segment.id}] Generating audio...")
                audio_result = p.generate_speech(segment)
                with results_lock:
                    generation_status["results"].append({
                        "segment_id": audio_result.segment_id,
                        "step": audio_result.step,
                        "success": audio_result.success,
                        "url": audio_result.url,
                        "local_path": audio_result.local_path,
                        "error": audio_result.error
                    })
                add_log(f"{'[OK]' if audio_result.success else '[ERROR]'} [{segment.id}] Audio: {'done' if audio_result.success else audio_result.error}")
                
                if generation_status["cancelled"]:
                    return
                
                # Image - check for screenshot OR direct composite images
                has_composite = (segment.needs_screenshot and segment.screenshot_urls) or \
                               (hasattr(segment, 'composite_image_urls') and segment.composite_image_urls)
                
                if has_composite and p.main_character_url:
                    add_log(f"[INFO] [{segment.id}] Creating composite...")
                    image_result = p.generate_composite_for_segment(segment)
                    if image_result is None:
                        image_result = p.generate_character_image(segment)
                else:
                    add_log(f"[INFO] [{segment.id}] Generating image...")
                    image_result = p.generate_character_image(segment)
                
                with results_lock:
                    generation_status["results"].append({
                        "segment_id": image_result.segment_id,
                        "step": image_result.step,
                        "success": image_result.success,
                        "url": image_result.url,
                        "local_path": image_result.local_path,
                        "error": image_result.error
                    })
                add_log(f"{'[OK]' if image_result.success else '[ERROR]'} [{segment.id}] Image: {'done' if image_result.success else image_result.error}")
                
                if generation_status["cancelled"]:
                    return
                
                # Video - use the image generated for THIS segment
                add_log(f"[INFO] [{segment.id}] Generating video...")
                audio_url = audio_result.url if audio_result.success else None
                image_url = image_result.url if image_result.success else p.main_character_url
                video_result = p.generate_video(segment, audio_url=audio_url, image_url=image_url)
                with results_lock:
                    generation_status["results"].append({
                        "segment_id": video_result.segment_id,
                        "step": video_result.step,
                        "success": video_result.success,
                        "url": video_result.url,
                        "local_path": video_result.local_path,
                        "error": video_result.error
                    })
                add_log(f"{'[OK]' if video_result.success else '[ERROR]'} [{segment.id}] Video: {'done' if video_result.success else video_result.error}")
                
                # Collect errors
                with results_lock:
                    for result in [audio_result, image_result, video_result]:
                        if not result.success:
                            generation_status["errors"].append(f"{segment.id}/{result.step}: {result.error}")
                    completed_count[0] += 1
                    generation_status["progress"] = int((completed_count[0] / len(segments)) * 100)
                
                add_log(f"[OK] {segment.id} COMPLETE ({completed_count[0]}/{len(segments)})")
            
            # Launch all segments in parallel with 5-second stagger
            with ThreadPoolExecutor(max_workers=len(segments)) as executor:
                futures = []
                for i, segment in enumerate(segments):
                    delay = i * 5  # 0s, 5s, 10s, 15s...
                    future = executor.submit(process_segment_parallel, segment, delay)
                    futures.append((segment.id, future))
                
                # Wait for all to complete
                for segment_id, future in futures:
                    try:
                        future.result()
                    except Exception as e:
                        add_log(f"[ERROR] {segment_id} error: {e}")
                    
            generation_status["progress"] = 100
            p.save_progress()
            
            if generation_status["cancelled"]:
                add_log("[WARN] Pipeline was cancelled")
            else:
                add_log("[OK] Pipeline complete! All materials generated.")
                
        except Exception as e:
            add_log(f"[ERROR] Pipeline error: {e}")
            generation_status["errors"].append(str(e))
        finally:
            generation_status["running"] = False
            generation_status["current_segment"] = None
            generation_status["current_step"] = None
    
    thread = threading.Thread(target=run_full)
    thread.start()
    
    return jsonify({
        "message": "Fully automated pipeline started",
        "segments": len(segments),
        "screenshots_needed": len([s for s in segments if s.needs_screenshot]),
        "total_steps": total_steps
    })


@app.route('/api/results')
def get_results():
    """Get all generation results from the pipeline."""
    p = get_pipeline()
    return jsonify({
        "summary": p.get_summary(),
        "results": generation_status["results"]
    })


@app.route('/api/regenerate/<segment_id>/<step>', methods=['POST'])
def regenerate_content(segment_id, step):
    """Regenerate a specific piece of content (audio, image, or video)."""
    if generation_status["running"]:
        return jsonify({"error": "Generation already in progress"}), 400
    
    # Find the segment
    p = get_pipeline()
    segment = None
    for s in p.config.get_segments():
        if s.id == segment_id:
            segment = s
            break
    
    if not segment:
        return jsonify({"error": f"Segment {segment_id} not found"}), 404
    
    if step not in ['audio', 'image', 'video']:
        return jsonify({"error": f"Invalid step: {step}"}), 400
    
    def run_regenerate():
        global generation_status
        generation_status["running"] = True
        generation_status["cancelled"] = False
        generation_status["current_segment"] = segment_id
        generation_status["current_step"] = step
        generation_status["progress"] = 0
        
        try:
            p = get_pipeline()
            add_log(f"[INFO] Regenerating {step} for {segment_id}...")
            
            result = None
            if step == 'audio':
                result = p.generate_speech(segment)
            elif step == 'image':
                has_composite = (segment.needs_screenshot and segment.screenshot_urls) or \
                               (hasattr(segment, 'composite_image_urls') and segment.composite_image_urls)
                if has_composite and p.main_character_url:
                    result = p.generate_composite_for_segment(segment)
                    if result is None:
                        result = p.generate_character_image(segment)
                else:
                    result = p.generate_character_image(segment)
            elif step == 'video':
                # Find existing audio and image URLs for this segment
                audio_url = None
                image_url = None
                for r in generation_status["results"]:
                    if r.get("segment_id") == segment_id and r.get("step") == "audio" and r.get("success"):
                        audio_url = r.get("url")
                    if r.get("segment_id") == segment_id and r.get("step") == "image" and r.get("success"):
                        image_url = r.get("url")
                # Fallback to main character if no image
                if not image_url:
                    image_url = p.main_character_url
                result = p.generate_video(segment, audio_url=audio_url, image_url=image_url)
            
            if result:
                # Update the result in the list
                result_dict = {
                    "segment_id": result.segment_id,
                    "step": result.step,
                    "success": result.success,
                    "url": result.url,
                    "local_path": result.local_path,
                    "error": result.error
                }
                
                # Replace existing result or add new one
                found = False
                for i, r in enumerate(generation_status["results"]):
                    if r.get("segment_id") == segment_id and r.get("step") == step:
                        generation_status["results"][i] = result_dict
                        found = True
                        break
                
                if not found:
                    generation_status["results"].append(result_dict)
                
                add_log(f"{'[OK]' if result.success else '[ERROR]'} {step} for {segment_id} {'completed' if result.success else 'failed'}")
            
            generation_status["progress"] = 100
            
        finally:
            generation_status["running"] = False
            generation_status["current_segment"] = None
            generation_status["current_step"] = None
    
    thread = threading.Thread(target=run_regenerate)
    thread.start()
    
    return jsonify({"message": f"Regenerating {step} for {segment_id}..."})


# ==================== IMAGE EDITING ENDPOINTS ====================

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Upload a file (screenshot) and return the fal URL."""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    try:
        p = get_pipeline()
        config = get_config()
        
        # Save the file locally first
        local_path = os.path.join(config.screenshots_dir, file.filename)
        file.save(local_path)
        
        # Upload to fal and get URL
        fal_url = p.upload_file(local_path)
        
        return jsonify({
            "success": True,
            "local_path": local_path,
            "url": fal_url,
            "filename": file.filename
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/edit/combine', methods=['POST'])
def combine_images():
    """
    Combine a screenshot with a character image.
    
    Request body:
    {
        "screenshot_url": "https://...",  // or local path
        "character_url": "https://...",   // URL of generated character
        "prompt": "Description of how to combine them",
        "name": "output_name"
    }
    """
    data = request.json or {}
    
    screenshot_url = data.get('screenshot_url')
    character_url = data.get('character_url')
    prompt = data.get('prompt', 'Combine these images naturally')
    name = data.get('name', 'composite')
    
    if not screenshot_url or not character_url:
        return jsonify({"error": "screenshot_url and character_url are required"}), 400
    
    if generation_status["running"]:
        return jsonify({"error": "Generation already in progress"}), 400
    
    def run_combine():
        global generation_status
        generation_status["running"] = True
        generation_status["current_step"] = "edit"
        generation_status["progress"] = 0
        
        try:
            p = get_pipeline()
            
            result = p.combine_screenshot_with_character(
                screenshot_path_or_url=screenshot_url,
                character_image_url=character_url,
                prompt=prompt,
                segment_id=name
            )
            
            generation_status["results"].append({
                "segment_id": result.segment_id,
                "step": result.step,
                "success": result.success,
                "url": result.url,
                "local_path": result.local_path,
                "error": result.error
            })
            
            generation_status["progress"] = 100
            
        finally:
            generation_status["running"] = False
            generation_status["current_step"] = None
    
    thread = threading.Thread(target=run_combine)
    thread.start()
    
    return jsonify({"message": "Image combination started", "name": name})


@app.route('/api/edit/image', methods=['POST'])
def edit_image():
    """
    Edit or combine multiple images using Nano Banana Pro Edit.
    
    Request body:
    {
        "prompt": "Description of the edit",
        "image_urls": ["https://...", "https://..."],  // Up to 14 images
        "name": "output_name",
        "aspect_ratio": "16:9"
    }
    """
    data = request.json or {}
    
    prompt = data.get('prompt')
    image_urls = data.get('image_urls', [])
    name = data.get('name', 'edited')
    aspect_ratio = data.get('aspect_ratio', '16:9')
    
    if not prompt:
        return jsonify({"error": "prompt is required"}), 400
    
    if not image_urls:
        return jsonify({"error": "At least one image_url is required"}), 400
    
    if len(image_urls) > 14:
        return jsonify({"error": "Maximum 14 images allowed"}), 400
    
    if generation_status["running"]:
        return jsonify({"error": "Generation already in progress"}), 400
    
    def run_edit():
        global generation_status
        generation_status["running"] = True
        generation_status["current_step"] = "edit"
        generation_status["progress"] = 0
        
        try:
            p = get_pipeline()
            
            result = p.edit_image(
                prompt=prompt,
                image_urls=image_urls,
                output_name=name,
                aspect_ratio=aspect_ratio
            )
            
            generation_status["results"].append({
                "segment_id": result.segment_id,
                "step": result.step,
                "success": result.success,
                "url": result.url,
                "local_path": result.local_path,
                "error": result.error
            })
            
            generation_status["progress"] = 100
            
        finally:
            generation_status["running"] = False
            generation_status["current_step"] = None
    
    thread = threading.Thread(target=run_edit)
    thread.start()
    
    return jsonify({"message": "Image editing started", "name": name})


@app.route('/api/screenshots')
def list_screenshots():
    """List all uploaded screenshots."""
    config = get_config()
    screenshots = []
    
    if os.path.exists(config.screenshots_dir):
        for filename in os.listdir(config.screenshots_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                screenshots.append({
                    "filename": filename,
                    "path": os.path.join(config.screenshots_dir, filename),
                    "url": f"/output/screenshots/{filename}"
                })
    
    return jsonify(screenshots)


@app.route('/api/composites')
def list_composites():
    """List all composite images."""
    config = get_config()
    composites = []
    
    if os.path.exists(config.composites_dir):
        for filename in os.listdir(config.composites_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                composites.append({
                    "filename": filename,
                    "path": os.path.join(config.composites_dir, filename),
                    "url": f"/output/composites/{filename}"
                })
    
    return jsonify(composites)


@app.route('/output/<path:filename>')
def serve_output(filename):
    """Serve files from the output directory."""
    config = get_config()
    return send_from_directory(config.output_dir, filename)


# Create templates directory and HTML file
TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), 'templates')
STATIC_DIR = os.path.join(os.path.dirname(__file__), 'static')

os.makedirs(TEMPLATES_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)


if __name__ == '__main__':
    # Ensure output directories exist
    config = get_config()
    
    print(f"""
    ============================================================
    
      Wan 2.6 Video Generation Pipeline
      
      Open http://localhost:5000 in your browser
    
    ============================================================
    """)
    
    app.run(debug=True, port=5000)

