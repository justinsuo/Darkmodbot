import ollama
import mss
import time
from PIL import Image
import io
import pydirectinput
import pyautogui
import re
import win32gui
import random
import hashlib
import numpy as np
from collections import deque

pydirectinput.FAILSAFE = False
pyautogui.FAILSAFE = False

# =============================================================================
# MEMORY - Smart tracking of behavior
# =============================================================================
class SmartMemory:
    def __init__(self):
        self.observations = deque(maxlen=50)
        self.visited_hashes = set()
        self.location_history = deque(maxlen=100)
        self.consecutive_pans = 0
        self.consecutive_forwards = 0
        self.stuck_counter = 0

    def add(self, desc, action, img_hash, moved):
        self.observations.append({'desc': desc, 'action': action, 'hash': img_hash})
        self.location_history.append(img_hash)

        if 'pan' in action and action not in ['pan up', 'pan down']:
            self.consecutive_pans += 1
            self.consecutive_forwards = 0
        elif action == 'move forward':
            self.consecutive_forwards += 1
            self.consecutive_pans = 0
        else:
            self.consecutive_pans = max(0, self.consecutive_pans - 1)

        if img_hash in self.visited_hashes:
            self.stuck_counter += 1
        else:
            self.visited_hashes.add(img_hash)
            self.stuck_counter = max(0, self.stuck_counter - 2)

        if moved:
            self.consecutive_forwards += 1

    def panning_too_much(self):
        return self.consecutive_pans >= 4

    def stuck(self):
        return self.stuck_counter > 6 or (len(self.location_history) >= 5 and len(set(list(self.location_history)[-5:])) <= 2)

    def summarize(self):
        recent = list(self.observations)[-5:]
        actions = " → ".join([o['action'] for o in recent])
        status = ""
        if self.stuck(): status = " [STUCK!]"
        elif self.panning_too_much(): status = " [PANNING TOO MUCH]"
        return f"{actions}{status}"

memory = SmartMemory()

# =============================================================================
# WINDOW FOCUS
# =============================================================================
def focus_darkmod_window():
    hwnd = None
    def callback(h, _):
        nonlocal hwnd
        title = win32gui.GetWindowText(h)
        if title and "dark mod" in title.lower():
            hwnd = h
            print(f"Found window: {title}")
            return False
    try:
        win32gui.EnumWindows(callback, None)
        if hwnd:
            win32gui.SetForegroundWindow(hwnd)
            time.sleep(0.1)
            w, h = pydirectinput.size()
            pydirectinput.click(w//2, h//2)
            return True
    except: pass
    return False

# =============================================================================
# SCREEN CAPTURE & HASH
# =============================================================================
def capture_screen():
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        sct_img = sct.grab(monitor)
        return Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")

def image_hash(img):
    small = img.resize((8,8), Image.Resampling.LANCZOS).convert("L")
    pixels = list(small.getdata())
    avg = sum(pixels) / len(pixels)
    bits = "".join("1" if p > avg else "0" for p in pixels)
    return hashlib.md5(bits.encode()).hexdigest()

# =============================================================================
# IMAGE ANALYSIS - Very lenient, favors movement
# =============================================================================
def analyze_image(img):
    gray = np.array(img.convert('L'))
    h, w = gray.shape

    center = gray[:, w//3:2*w//3]
    left   = gray[:, :w//3]
    right  = gray[:, 2*w//3:]

    top    = gray[:h//3, :]
    bottom = gray[2*h//3:, :]

    def stats(area):
        if area.size == 0: return {'var': 0, 'edges': 0}
        var = np.var(area)
        edges = np.mean(np.abs(np.diff(area, axis=0))) + np.mean(np.abs(np.diff(area, axis=1)))
        return {'var': var, 'edges': edges}

    c = stats(center)
    l = stats(left)
    r = stats(right)
    t = stats(top)
    b = stats(bottom)

    looking_up   = t['var'] < b['var'] * 0.5
    looking_down = b['var'] < t['var'] * 0.5

    # Very lenient: only block if extremely uniform
    blocked = lambda s: s['var'] < 180 and s['edges'] < 20

    return {
        'obstacles': {'left': blocked(l), 'center': blocked(c), 'right': blocked(r)},
        'looking_up': looking_up,
        'looking_down': looking_down,
        'open_space': c['var'] > 400 or c['edges'] > 30,
        'very_clear': c['var'] > 600
    }

# =============================================================================
# AI PROMPT - Direct and aggressive
# =============================================================================
def ask_ai(img, memory_summary, last_action, img_analysis):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    base64_img = base64.b64encode(buffered.getvalue()).decode()

    prompt = f"""
You are an AGGRESSIVE exploration robot in a 3D game.
GOAL: MOVE FORWARD and explore new areas as fast as possible.

Memory: {memory_summary}
Last action: {last_action}
Obstacles detected: {img_analysis['obstacles']}

Answer these 3 questions:
1. Can I walk forward right now without hitting a wall in the next 1 second?
   → YES or NO
2. What do I see directly ahead?
   → open path / wall / door / stairs / object
3. Is my view at eye level?
   → eye level / looking up / looking down

RULES:
- Say YES to walking forward unless a wall is IMMEDIATELY in front
- Prefer "move forward" aggressively
- Only pan to fix view angle or find a path
- Never pan more than twice in a row

Output exactly:
CanWalk: YES/NO
View: eye level/looking up/looking down
Center: open/wall/door/stairs/object
Decision: move forward / pan left / pan right / pan up / pan down / turn around / move forward and jump / approach and interact
"""

    resp = ollama.generate(model='llava:13b', prompt=prompt, images=[base64_img])
    return resp['response']

# =============================================================================
# ACTION DECISION - Smart + aggressive
# =============================================================================
def decide_action(ai_text, img_analysis, memory):
    # Fix view first
    if img_analysis['looking_up']:   return 'pan down'
    if img_analysis['looking_down']: return 'pan up'

    # Stop panning madness
    if memory.panning_too_much():
        print("FORCING FORWARD - too much panning!")
        return 'move forward'

    # Trust AI if it says yes
    if "CanWalk: YES" in ai_text.upper():
        return 'move forward'

    # Very clear path → go
    if img_analysis['very_clear']:
        return 'move forward'

    # Open space → go
    if img_analysis['open_space'] and not img_analysis['obstacles']['center']:
        return 'move forward'

    # Extract AI decision
    match = re.search(r'Decision:\s*(.+)', ai_text, re.I)
    if match:
        act = match.group(1).strip().lower()
        valid = ['move forward','pan left','pan right','pan up','pan down','turn around','move forward and jump','approach and interact']
        for v in valid:
            if v in act:
                return v

    # Default: try to move or turn
    if not img_analysis['obstacles']['center']:
        return 'move forward'
    return 'pan left' if random.random() < 0.5 else 'pan right'

# =============================================================================
# ACTION EXECUTION
# =============================================================================
def do(action):
    focus_darkmod_window()

    mapping = {
        'move forward': ('w', 0.8),
        'move backward': ('s', 0.6),
        'move left': ('a', 0.5),
        'move right': ('d', 0.5),
        'pan left': (-380, 0),
        'pan right': (380, 0),
        'pan up': (0, -380),
        'pan down': (0, 380),
        'interact': ('f', 0.1),
    }

    if action in ['move forward','move backward','move left','move right','interact']:
        key, dur = mapping[action]
        pydirectinput.keyDown(key)
        time.sleep(dur)
        pydirectinput.keyUp(key)
        print(f"→ {action}")

    elif 'pan' in action:
        dx, dy = mapping[action]
        pydirectinput.moveRel(dx, dy, relative=True)
        time.sleep(0.1)
        w, h = pydirectinput.size()
        pydirectinput.moveTo(w//2, h//2)
        print(f"→ {action}")

    elif action == 'turn around':
        do('pan left')
        time.sleep(0.1)
        do('pan left')

    elif action == 'move forward and jump':
        pydirectinput.keyDown('w')
        time.sleep(0.3)
        pydirectinput.press('space')
        time.sleep(0.4)
        pydirectinput.keyUp('w')

    elif action == 'approach and interact':
        do('move forward')
        time.sleep(0.3)
        pydirectinput.press('f')

# =============================================================================
# MAIN LOOP
# =============================================================================
print("\n" + "="*70)
print("AGGRESSIVE EXPLORATION BOT - READY")
print("="*70 + "\n")

step = 0
last_action = "none"

while True:
    try:
        step += 1
        print(f"\n{'='*60} STEP {step} {'='*60}")

        img = capture_screen()
        pre_hash = image_hash(img)
        analysis = analyze_image(img)

        print(f"Obstacles: {analysis['obstacles']} | View: {'Up' if analysis['looking_up'] else 'Down' if analysis['looking_down'] else 'Level'}")

        ai_response = ask_ai(img, memory.summarize(), last_action, analysis)
        print(f"AI says: {ai_response.splitlines()[0] if ai_response else '...'}")

        action = decide_action(ai_response, analysis, memory)
        print(f"DECISION → {action.upper()}")

        do(action)
        last_action = action

        # Collision check
        time.sleep(0.3)
        post_img = capture_screen()
        post_hash = image_hash(post_img)

        moved = post_hash != pre_hash
        if action == 'move forward' and not moved:
            print("COLLISION! Backing up + turning")
            do('move backward')
            do(random.choice(['turn around', 'pan left', 'pan right']))

        memory.add("exploring", action, post_hash, moved)

        # After any pan → try to move forward
        if 'pan' in action and action not in ['pan up', 'pan down']:
            time.sleep(0.2)
            new_analysis = analyze_image(capture_screen())
            if not new_analysis['obstacles']['center']:
                print("Path clear after pan → MOVING FORWARD")
                do('move forward')

        time.sleep(0.4)

    except KeyboardInterrupt:
        print("\n\nStopped by user.")
        break
    except Exception as e:
        print(f"Error: {e}")
        import traceback; traceback.print_exc()
        time.sleep(1)

print("Exploration ended.")
