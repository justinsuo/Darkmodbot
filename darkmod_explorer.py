import ollama
import mss
import mss.tools
import base64
import time
from PIL import Image
import io
import pydirectinput
import pyautogui
import re
import win32gui
import win32con
import random
import hashlib
import numpy as np
from collections import deque

pydirectinput.FAILSAFE = False
pyautogui.FAILSAFE = False

# Enhanced Memory System with aggressive exploration bias
class SmartMemory:
    def __init__(self, max_size=50):
        self.observations = deque(maxlen=max_size)
        self.visited_hashes = set()
        self.location_history = deque(maxlen=100)
        self.stuck_counter = 0
        self.last_successful_direction = None
        self.exploration_goals = deque(maxlen=10)
        self.obstacle_map = {}
        self.interest_points = []
        self.consecutive_pans = 0  # Track excessive panning
        self.consecutive_forwards = 0  # Track forward movement
        self.last_actions = deque(maxlen=5)  # Track recent actions
        
    def add_observation(self, desc, action, img_hash, success=True):
        self.observations.append({
            'description': desc,
            'action': action,
            'hash': img_hash,
            'timestamp': time.time(),
            'success': success
        })
        self.location_history.append(img_hash)
        self.last_actions.append(action)
        
        # Track action patterns
        if 'pan' in action:
            self.consecutive_pans += 1
            self.consecutive_forwards = 0
        elif action == 'move forward':
            self.consecutive_forwards += 1
            self.consecutive_pans = 0
        else:
            self.consecutive_pans = 0
            self.consecutive_forwards = 0
        
        if img_hash in self.visited_hashes:
            self.stuck_counter += 1
        else:
            self.visited_hashes.add(img_hash)
            self.stuck_counter = max(0, self.stuck_counter - 2)  # Reward progress
            
        if success:
            self.last_successful_direction = action
    
    def is_panning_too_much(self):
        """Detect if agent is panning excessively instead of moving"""
        return self.consecutive_pans >= 3
    
    def is_stuck(self):
        if len(self.location_history) < 5:
            return False
        recent = list(self.location_history)[-5:]
        return len(set(recent)) <= 2 or self.stuck_counter > 4
    
    def should_force_forward(self):
        """Encourage forward movement"""
        # If we haven't moved forward in a while, force it
        recent_actions = list(self.last_actions)
        forward_count = sum(1 for a in recent_actions if a == 'move forward')
        return forward_count < 2 and not self.is_stuck()
    
    def get_unexplored_direction(self):
        actions = ['pan left', 'pan right', 'turn around']
        return random.choice(actions)
    
    def summarize(self):
        if not self.observations:
            return "No previous observations."
        
        recent = list(self.observations)[-5:]
        summary = " | ".join([f"{obs['action']}: {obs['description'][:50]}" for obs in recent])
        
        status = ""
        if self.is_stuck():
            status = " [STUCK - Need new strategy]"
        elif self.is_panning_too_much():
            status = " [PANNING TOO MUCH - Should move forward]"
        elif self.last_successful_direction:
            status = f" [Last success: {self.last_successful_direction}]"
            
        return summary + status

memory = SmartMemory()

# Spatial awareness and mapping
class SpatialMap:
    def __init__(self):
        self.grid = {}
        self.current_position = (0, 0)
        self.current_heading = 0
        self.forward_bias = 0.7  # Bias toward forward movement
        
    def update_heading(self, action):
        if action == 'pan left':
            self.current_heading = (self.current_heading - 45) % 360
        elif action == 'pan right':
            self.current_heading = (self.current_heading + 45) % 360
        elif action == 'turn around':
            self.current_heading = (self.current_heading + 180) % 360
            
    def estimate_move(self, action):
        if 'forward' in action:
            rad = np.radians(self.current_heading)
            dx = int(np.cos(rad))
            dy = int(np.sin(rad))
            self.current_position = (
                self.current_position[0] + dx,
                self.current_position[1] + dy
            )
        elif 'backward' in action:
            rad = np.radians(self.current_heading)
            dx = int(np.cos(rad))
            dy = int(np.sin(rad))
            self.current_position = (
                self.current_position[0] - dx,
                self.current_position[1] - dy
            )
            
    def mark_obstacle(self, direction):
        pos_key = f"{self.current_position}_{self.current_heading}"
        self.grid[pos_key] = 'obstacle'
        
    def get_exploration_target(self):
        directions = [0, 45, 90, 135, 180, 225, 270, 315]
        unexplored = []
        
        for heading in directions:
            pos_key = f"{self.current_position}_{heading}"
            if pos_key not in self.grid:
                unexplored.append(heading)
                
        if unexplored:
            target = random.choice(unexplored)
            diff = (target - self.current_heading) % 360
            if diff > 180:
                return 'pan left'
            elif diff > 0:
                return 'pan right'
        return None

spatial_map = SpatialMap()

def list_all_windows():
    windows = []
    def enum_cb(hwnd, results):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title:
                results.append(title)
    win32gui.EnumWindows(enum_cb, windows)
    return windows

print("Diagnostic: All open window titles:")
print("\n".join(list_all_windows()))
print("\nLooking for Dark Mod window...\n")

def focus_darkmod_window():
    found_hwnd = None
    def enum_cb(hwnd, _):
        nonlocal found_hwnd
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title and "dark mod" in title.lower():
                found_hwnd = hwnd
                print(f"Found The Dark Mod window: '{title}'")
                return False
    try:
        win32gui.EnumWindows(enum_cb, None)
        if found_hwnd:
            win32gui.SetForegroundWindow(found_hwnd)
            time.sleep(0.1)
            screen_width, screen_height = pydirectinput.size()
            pydirectinput.click(screen_width // 2, screen_height // 2)
            return True
        else:
            print("No Dark Mod window found.")
            return False
    except Exception as e:
        print(f"Focus error: {e}")
        return False

def capture_screen():
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        sct_img = sct.grab(monitor)
        img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
        return img

def image_hash(img):
    img = img.resize((8, 8), Image.Resampling.LANCZOS).convert("L")
    pixels = list(img.getdata())
    avg = sum(pixels) / len(pixels)
    bits = "".join("1" if p > avg else "0" for p in pixels)
    return hashlib.md5(bits.encode()).hexdigest()

def advanced_image_analysis(img):
    """Enhanced image processing - more lenient on obstacles"""
    img_array = np.array(img.convert('L'))
    h, w = img_array.shape
    
    # Analyze different regions
    top_third = img_array[:h//3, :]
    middle_third = img_array[h//3:2*h//3, :]
    bottom_third = img_array[2*h//3:, :]
    
    # Split horizontally
    left = img_array[:, :w//3]
    center = img_array[:, w//3:2*w//3]
    right = img_array[:, 2*w//3:]
    
    def analyze_region(region):
        sobelx = np.abs(np.diff(region, axis=1))
        sobely = np.abs(np.diff(region, axis=0))
        
        min_h = min(sobelx.shape[0], sobely.shape[0])
        min_w = min(sobelx.shape[1], sobely.shape[1])
        sobelx = sobelx[:min_h, :min_w]
        sobely = sobely[:min_h, :min_w]
        
        edges = np.sqrt(np.square(sobelx) + np.square(sobely))
        
        return {
            'edge_density': np.mean(edges),
            'variance': np.var(region),
            'mean_brightness': np.mean(region),
            'has_detail': np.std(region) > 20,
            'is_dark': np.mean(region) < 50
        }
    
    analysis = {
        'left': analyze_region(left),
        'center': analyze_region(center),
        'right': analyze_region(right),
        'top': analyze_region(top_third),
        'middle': analyze_region(middle_third),
        'bottom': analyze_region(bottom_third)
    }
    
    # Determine view state
    looking_up = (analysis['top']['variance'] < analysis['bottom']['variance'] * 0.5 and
                  analysis['top']['mean_brightness'] > analysis['bottom']['mean_brightness'])
    looking_down = (analysis['bottom']['variance'] < analysis['top']['variance'] * 0.5)
    
    # More lenient obstacle detection - prefer movement over caution
    obstacle_threshold_variance = 200  # Lower = more lenient
    obstacle_threshold_edges = 20  # Lower = more lenient
    
    obstacles = {
        'left': (analysis['left']['variance'] < obstacle_threshold_variance and
                 analysis['left']['edge_density'] < obstacle_threshold_edges),
        'center': (analysis['center']['variance'] < obstacle_threshold_variance and
                   analysis['center']['edge_density'] < obstacle_threshold_edges),
        'right': (analysis['right']['variance'] < obstacle_threshold_variance and
                  analysis['right']['edge_density'] < obstacle_threshold_edges)
    }
    
    # Detect interesting features
    interesting_features = {
        'potential_door': analysis['center']['has_detail'] and 40 < analysis['center']['edge_density'] < 80,
        'potential_stairs': analysis['bottom']['edge_density'] > 60 and analysis['bottom']['variance'] > 400,
        'open_space': analysis['center']['variance'] > 300 or analysis['center']['edge_density'] > 25,  # More lenient
        'very_clear_path': analysis['center']['variance'] > 500 and analysis['center']['edge_density'] > 40
    }
    
    return {
        'obstacles': obstacles,
        'looking_up': looking_up,
        'looking_down': looking_down,
        'features': interesting_features,
        'analysis': analysis
    }

def analyze_screen_with_ai(img, memory_summary, last_action, image_analysis):
    """MUCH BETTER AI PROMPTING - Ask specific, direct questions"""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    prompt = f"""
You are an exploration AI in a first-person 3D game. Your PRIMARY GOAL is to MOVE FORWARD and EXPLORE as much as possible.
CONTEXT:
- Previous actions: {memory_summary}
- Last action: {last_action}
- Technical analysis: {image_analysis['obstacles']}
CRITICAL QUESTIONS - Answer each one clearly:
1. IS THERE ANYTHING DIRECTLY IN FRONT OF ME RIGHT NOW?
   - Look at the CENTER of the image
   - Is there a wall, object, or obstacle RIGHT IN FRONT blocking forward movement?
   - Answer: YES (blocked) or NO (clear)
2. HOW CLOSE IS THE NEAREST OBSTACLE IN FRONT?
   - Very close (within 1 meter - would hit immediately)
   - Medium distance (2-5 meters - can move forward safely)
   - Far away (5+ meters - plenty of room)
   - No obstacle visible (completely clear)
3. WHAT DO I SEE IN THE CENTER OF MY VIEW?
   - Describe ONLY what's directly ahead
   - Is it: open hallway, room, doorway, wall, corner, stairs, or something else?
4. CAN I WALK FORWARD RIGHT NOW?
   - Answer: YES or NO
   - If NO, why not? (wall/object/drop-off)
5. WHAT'S TO MY LEFT AND RIGHT?
   - LEFT: clear path / wall / object
   - RIGHT: clear path / wall / object
6. IS MY VIEW ANGLE CORRECT?
   - Am I looking at eye level (can see floor and ahead)?
   - Am I looking up at ceiling/sky?
   - Am I looking down at floor?
EXPLORATION PRIORITY (in order):
1. If center is clear → MOVE FORWARD (explore new areas)
2. If view angle wrong → Fix it (pan up/down)
3. If center blocked but left/right clear → Turn toward open side
4. If completely stuck → Turn around
IMPORTANT RULES:
- PREFER "move forward" whenever center looks even remotely passable
- Don't be overly cautious - if there's space, GO
- Only suggest panning if view angle is wrong OR path is clearly blocked
- Avoid staying in one place - keep moving
OUTPUT FORMAT:
Direct Answer 1: [YES/NO - Is center blocked?]
Direct Answer 2: [Distance to nearest obstacle]
Direct Answer 3: [What I see ahead]
Direct Answer 4: [YES/NO - Can walk forward?]
View Angle: [eye level/looking up/looking down]
Left Side: [clear/blocked]
Right Side: [clear/blocked]
Decision: [ONE action: move forward / pan left / pan right / pan up / pan down / turn around / move forward and jump / approach and interact]
Reasoning: [One sentence explaining why]
"""
    
    response = ollama.generate(
        model='llava:13b',
        prompt=prompt,
        images=[base64_image]
    )
    
    return response['response']

def extract_action(decision_text):
    """Extract action with priority order"""
    actions = re.findall(r'(move forward and jump|crouch and move forward|approach and interact|move forward|move left|move right|move backward|pan left|pan right|pan up|pan down|turn around)', decision_text.lower())
    if actions:
        return actions[0]
    return None

def parse_ai_response(ai_response):
    """Parse the structured AI response"""
    can_walk = False
    center_blocked = True
    
    # Look for direct answers
    if re.search(r'Direct Answer 1:.*NO', ai_response, re.IGNORECASE):
        center_blocked = False
    if re.search(r'Direct Answer 4:.*YES', ai_response, re.IGNORECASE):
        can_walk = True
    
    # Check for clear path indicators
    if re.search(r'(clear|open|hallway|room|space|passage)', ai_response, re.IGNORECASE):
        can_walk = True
    
    # Check for obstacles
    if re.search(r'(wall|blocked|obstacle|close|immediately)', ai_response, re.IGNORECASE):
        can_walk = False
        center_blocked = True
    
    return can_walk, center_blocked

def smart_action_selection(decision_text, image_analysis, memory, ai_response):
    """AGGRESSIVE action selection - prioritize forward movement"""
    
    # Parse AI response for better understanding
    can_walk, center_blocked_ai = parse_ai_response(ai_response)
    
    action = extract_action(decision_text)
    
    # FORCE FORWARD MOVEMENT if panning too much
    if memory.is_panning_too_much():
        print("🚀 Override: Panned too much, FORCING forward movement")
        return 'move forward'
    
    # Override if view angle is wrong
    if image_analysis['looking_up']:
        print("👁️ Override: Looking up, correcting view")
        return 'pan down'
    if image_analysis['looking_down']:
        print("👁️ Override: Looking down, correcting view")
        return 'pan up'
    
    # AGGRESSIVE: If AI says we can walk, DO IT
    if can_walk and action != 'move forward':
        print("🚀 Override: AI says path is clear, moving forward")
        return 'move forward'
    
    # AGGRESSIVE: If image analysis shows open space, move forward
    if image_analysis['features']['open_space'] and not image_analysis['obstacles']['center']:
        print("🚀 Override: Open space detected, moving forward")
        return 'move forward'
    
    # AGGRESSIVE: If very clear path, definitely move
    if image_analysis['features']['very_clear_path']:
        print("🚀 Override: Very clear path, definitely moving forward")
        return 'move forward'
    
    # If stuck, aggressive recovery
    if memory.is_stuck():
        print("🔄 Override: Stuck detected, aggressive recovery")
        return 'turn around'
    
    # If action is pan but we should move forward
    if action in ['pan left', 'pan right'] and memory.should_force_forward():
        if not image_analysis['obstacles']['center']:
            print("🚀 Override: Replacing pan with forward movement")
            return 'move forward'
    
    # Only block forward if DEFINITELY blocked
    if action == 'move forward':
        # Only stop if BOTH AI and image analysis say blocked
        if image_analysis['obstacles']['center'] and center_blocked_ai:
            print("🛑 Override: Center definitely blocked, turning")
            if not image_analysis['obstacles']['left']:
                return 'pan left'
            elif not image_analysis['obstacles']['right']:
                return 'pan right'
            else:
                return 'turn around'
        else:
            print("✅ Forward movement approved")
            return 'move forward'
    
    # Prioritize interesting features
    if image_analysis['features']['potential_door']:
        print("🚪 Override: Door detected")
        return 'approach and interact'
    
    if image_analysis['features']['potential_stairs']:
        print("🪜 Override: Stairs detected")
        return 'move forward and jump'
    
    # If no action, default to forward if possible
    if not action:
        if not image_analysis['obstacles']['center']:
            print("🚀 Default: Moving forward")
            return 'move forward'
        else:
            suggested = spatial_map.get_exploration_target()
            if suggested:
                return suggested
            return 'pan left'
    
    return action

def perform_action(decision, use_pydirectinput=True):
    """Execute action with better timing"""
    action_map = {
        'move forward': 'w',
        'move left': 'a',
        'move right': 'd',
        'move backward': 's',
        'pan left': (-350, 0),  # Slightly less aggressive panning
        'pan right': (350, 0),
        'pan up': (0, -350),
        'pan down': (0, 350),
        'interact': 'f',
        'jump': 'space',
        'crouch': 'ctrlleft'
    }
    
    if not focus_darkmod_window():
        print("Skipping action due to focus failure")
        return False
    
    try:
        if decision in action_map:
            if isinstance(action_map[decision], str):
                key = action_map[decision]
                duration = 0.8 if key == 'w' else 0.6  # Longer forward movement
                
                if use_pydirectinput:
                    pydirectinput.keyDown(key)
                    time.sleep(duration)
                    pydirectinput.keyUp(key)
                else:
                    pyautogui.keyDown(key)
                    time.sleep(duration)
                    pyautogui.keyUp(key)
                print(f"✓ Pressed key: {key} for {duration}s")
            else:
                dx, dy = action_map[decision]
                if use_pydirectinput:
                    pydirectinput.moveRel(dx, dy, relative=True)
                else:
                    pyautogui.moveRel(dx, dy)
                print(f"✓ Moved mouse: dx={dx}, dy={dy}")
                
                screen_width, screen_height = pydirectinput.size()
                pydirectinput.moveTo(screen_width // 2, screen_height // 2)
                
        elif decision == 'turn around':
            perform_action('pan left')
            time.sleep(0.1)
            perform_action('pan left')
            
        elif decision == 'move forward and jump':
            pydirectinput.keyDown('w')
            time.sleep(0.3)
            pydirectinput.press('space')
            time.sleep(0.4)
            pydirectinput.keyUp('w')
            
        elif decision == 'crouch and move forward':
            pydirectinput.press('ctrlleft')
            time.sleep(0.1)
            perform_action('move forward')
            time.sleep(0.1)
            pydirectinput.press('ctrlleft')
            
        elif decision == 'approach and interact':
            perform_action('move forward')
            time.sleep(0.3)
            pydirectinput.press('f')
            time.sleep(0.5)
        
        spatial_map.update_heading(decision)
        spatial_map.estimate_move(decision)
        
        return True
        
    except Exception as e:
        print(f"Action error: {e}")
        return False

# Main exploration loop
print("\n" + "="*60)
print("🚀 AGGRESSIVE EXPLORATION AI - Starting...")
print("="*60 + "\n")
last_action = 'none'
action_count = 0
consecutive_collisions = 0
while True:
    try:
        action_count += 1
        print(f"\n{'='*60}")
        print(f"STEP {action_count}")
        print(f"{'='*60}")
        
        # Capture pre-action state
        pre_img = capture_screen()
        pre_hash = image_hash(pre_img)
        
        # Advanced image analysis
        image_analysis = advanced_image_analysis(pre_img)
        print(f"\n📊 Image Analysis:")
        print(f" Obstacles: L={image_analysis['obstacles']['left']}, C={image_analysis['obstacles']['center']}, R={image_analysis['obstacles']['right']}")
        print(f" View: {'Looking Up' if image_analysis['looking_up'] else 'Looking Down' if image_analysis['looking_down'] else 'Eye Level'}")
        print(f" Features: {image_analysis['features']}")
        
        # Get memory summary
        memory_summary = memory.summarize()
        print(f"\n🧠 Memory: {memory_summary}")
        print(f" Stuck Counter: {memory.stuck_counter}")
        print(f" Visited Locations: {len(memory.visited_hashes)}")
        print(f" Consecutive Pans: {memory.consecutive_pans}")
        print(f" Consecutive Forwards: {memory.consecutive_forwards}")
        
        # AI analysis with better prompting
        print(f"\n🤖 Analyzing scene with AI...")
        ai_response = analyze_screen_with_ai(pre_img, memory_summary, last_action, image_analysis)
        print(f"\n📝 AI Response:\n{ai_response}\n")
        
        # Smart action selection
        action = smart_action_selection(ai_response, image_analysis, memory, ai_response)
        
        if not action:
            print("⚠️ No valid action determined, defaulting to forward")
            action = 'move forward' if not image_analysis['obstacles']['center'] else 'pan left'
        
        print(f"\n🎯 Selected Action: {action.upper()}")
        
        # Execute action
        success = perform_action(action, use_pydirectinput=True)
        
        if not success:
            print("⚠️ Action failed, trying pyautogui")
            success = perform_action(action, use_pydirectinput=False)
        
        # Post-action verification
        time.sleep(0.3)
        post_img = capture_screen()
        post_hash = image_hash(post_img)
        
        # Collision detection
        if action == 'move forward' and pre_hash == post_hash:
            consecutive_collisions += 1
            print(f"\n⚠️ COLLISION DETECTED #{consecutive_collisions}")
            spatial_map.mark_obstacle('forward')
            memory.add_observation("Collision detected", action, pre_hash, success=False)
            
            # Aggressive recovery
            print("🔄 Executing recovery...")
            perform_action('move backward')
            time.sleep(0.2)
            
            # More varied recovery
            if consecutive_collisions >= 3:
                print("🔄 Multiple collisions, turning around")
                perform_action('turn around')
                consecutive_collisions = 0
            else:
                recovery_action = random.choice(['pan left', 'pan right'])
                perform_action(recovery_action)
            
            last_action = f"{action} -> collision recovery"
        else:
            consecutive_collisions = 0
            desc_match = re.search(r'Direct Answer 3:\s*(.+)', ai_response, re.IGNORECASE)
            if not desc_match:
                desc_match = re.search(r'Description:\s*(.+)', ai_response, re.IGNORECASE)
            desc = desc_match.group(1)[:100] if desc_match else 'N/A'
            memory.add_observation(desc, action, post_hash, success=True)
            last_action = action
            
            # AGGRESSIVE: After panning, immediately try to move forward
            if 'pan' in action and action not in ['pan up', 'pan down']:
                time.sleep(0.2)
                new_img = capture_screen()
                new_analysis = advanced_image_analysis(new_img)
                
                # More lenient check
                if not new_analysis['obstacles']['center'] or new_analysis['features']['open_space']:
                    print("✅ Moving forward after pan")
                    perform_action('move forward')
                    time.sleep(0.3)
                    post_img = capture_screen()
                    post_hash = image_hash(post_img)
                    memory.add_observation("Moved forward after pan", 'move forward', post_hash, success=True)
                    last_action = 'move forward'
        
        # Periodic status
        if action_count % 10 == 0:
            print(f"\n{'='*60}")
            print(f"📈 EXPLORATION STATS (Step {action_count})")
            print(f" Unique locations visited: {len(memory.visited_hashes)}")
            print(f" Current position estimate: {spatial_map.current_position}")
            print(f" Current heading: {spatial_map.current_heading}°")
            print(f" Stuck counter: {memory.stuck_counter}")
            forward_ratio = memory.consecutive_forwards / max(1, memory.consecutive_forwards + memory.consecutive_pans)
            print(f" Movement ratio: {forward_ratio:.1%} forward")
            print(f"{'='*60}\n")
        
        # Shorter delay for more action
        time.sleep(0.4)
        
    except KeyboardInterrupt:
        print("\n\n🛑 Exploration stopped by user")
        print(f"Total steps: {action_count}")
        print(f"Unique locations: {len(memory.visited_hashes)}")
        break
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        time.sleep(1)

print("\n" + "="*60)
print("Exploration session ended")
print("="*60)
