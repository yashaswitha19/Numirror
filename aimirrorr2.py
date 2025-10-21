import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
import io
import threading
import time
import atexit
import queue
import subprocess
import os
import platform
import psutil
from colorthief import ColorThief
from transformers import YolosFeatureExtractor, YolosForObjectDetection
import speech_recognition as sr
from onnxruntime_genai import Model, Tokenizer, Generator, GeneratorParams
from fer import FER # Replace Keras imports with FER
import warnings
# Suppress warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
from transformers.utils import logging
logging.set_verbosity_error()
warnings.filterwarnings("ignore")


class StreamlitCompatibleSpeechManager:
    """
    A speech manager designed specifically for Streamlit that avoids
    the 'run loop already started' error by using system commands
    """
    def __init__(self):
        self.queue = queue.Queue()
        self.worker_thread = None
        self.running = False
        self.system = platform.system().lower()
        self.speaking = False
        self.current_process = None # Track current TTS process
        self.stop_requested = False # Flag to request stopping
        self.process_lock = threading.Lock() # Lock for process safety
        
    def start(self):
        """Start the speech worker thread"""
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        print("Speech Manager started successfully.")

    def _worker(self):
        """Worker thread that processes speech requests"""
        while self.running:
            try:
                text = self.queue.get(timeout=1.0)
                if text is None:
                    break
                
                # Reset stop flag for new speech
                self.stop_requested = False
                
                # Wait if currently speaking (but check for stop requests)
                while self.speaking and self.running and not self.stop_requested:
                    time.sleep(0.1)
                
                if self.running and not self.stop_requested:
                    self.speaking = True
                    print(f"Now speaking: '{text[:50]}...'")
                    self._speak_system(text)
                    self.speaking = False
                    print(f"Finished speaking: '{text[:50]}...'")
                else:
                    print(f"Speech cancelled: '{text[:50]}...'")
                
                self.queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Speech Worker Error: {e}")
                self.speaking = False
                self._cleanup_process()
                try:
                    self.queue.task_done()
                except:
                    pass

    def _speak_system(self, text):
        """Use system TTS commands instead of pyttsx3"""
        try:
            # Clean the text for speech
            clean_text = text.replace('"', "'").replace("`", "'").replace('\n', ' ').strip()
            
            if not clean_text or self.stop_requested:
                return
            
            with self.process_lock:
                if self.system == "windows":
                    # Use Windows SAPI with process creation flags to make it easier to kill
                    escaped_text = clean_text.replace("'", "''")
                    cmd = f'''powershell -Command "Add-Type -AssemblyName System.Speech; $speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; $speak.Rate = 0; $speak.Speak('{escaped_text}')"'''
                    
                    # Use CREATE_NEW_PROCESS_GROUP to make it easier to terminate
                    self.current_process = subprocess.Popen(
                        cmd, shell=True, stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE, text=True,
                        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if self.system == "windows" else 0
                    )
                    
                elif self.system == "darwin": # macOS
                    # Use macOS say command with process group
                    self.current_process = subprocess.Popen(
                        ["say", clean_text], stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE, text=True,
                        preexec_fn=os.setsid if hasattr(os, 'setsid') else None
                    )
                    
                elif self.system == "linux":
                    # Try different Linux TTS options with process groups
                    tts_commands = [
                        ["espeak", "-s", "150", clean_text],
                        ["spd-say", "-r", "-10", clean_text],
                    ]
                    
                    for cmd in tts_commands:
                        try:
                            self.current_process = subprocess.Popen(
                                cmd, stdout=subprocess.PIPE, 
                                stderr=subprocess.PIPE, text=True,
                                preexec_fn=os.setsid if hasattr(os, 'setsid') else None
                            )
                            break
                        except FileNotFoundError:
                            continue
                
                # Monitor process with frequent stop checks
                if self.current_process:
                    try:
                        start_time = time.time()
                        max_speech_time = 30 # Maximum 30 seconds per speech
                        
                        # Poll the process very frequently while checking for stop requests
                        while self.current_process.poll() is None:
                            if self.stop_requested:
                                print("Stop requested, terminating TTS process")
                                self._terminate_process()
                                return
                            
                            # Timeout protection
                            if time.time() - start_time > max_speech_time:
                                print("TTS timeout, terminating process")
                                self._terminate_process()
                                return
                                
                            time.sleep(0.05) # Check every 50ms for very responsive stopping
                        
                        if not self.stop_requested:
                            # Process completed naturally
                            stdout, stderr = self.current_process.communicate(timeout=1)
                            if self.current_process.returncode != 0 and stderr:
                                print(f"TTS Error: {stderr}")
                        
                    except subprocess.TimeoutExpired:
                        print("TTS process communication timeout")
                        self._terminate_process()
                    except Exception as e:
                        print(f"Error monitoring TTS process: {e}")
                        self._terminate_process()
                    finally:
                        self.current_process = None
                        
        except Exception as e:
            print(f"TTS Error: {e}")
            self._cleanup_process()

    def _terminate_process(self):
        """Forcefully terminate the current TTS process and all child processes"""
        if self.current_process:
            try:
                # Get process PID before termination
                pid = self.current_process.pid
                
                # Try to kill all child processes first
                try:
                    parent = psutil.Process(pid)
                    children = parent.children(recursive=True)
                    
                    # Kill all child processes
                    for child in children:
                        try:
                            child.terminate()
                        except psutil.NoSuchProcess:
                            pass
                    
                    # Wait for children to terminate
                    gone, alive = psutil.wait_procs(children, timeout=1)
                    
                    # Force kill any remaining children
                    for child in alive:
                        try:
                            child.kill()
                        except psutil.NoSuchProcess:
                            pass
                            
                except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                    print(f"Could not access child processes: {e}")
                
                # Now terminate the main process
                self.current_process.terminate()
                try:
                    self.current_process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    # Force kill the main process
                    self.current_process.kill()
                    self.current_process.wait()
                
                # Additional cleanup for Windows SAPI
                if self.system == "windows":
                    self._kill_windows_sapi_processes()
                    
                print("TTS process and children terminated")
            except Exception as e:
                print(f"Error terminating TTS process: {e}")
            finally:
                self.current_process = None

    def _kill_windows_sapi_processes(self):
        """Kill Windows SAPI related processes"""
        try:
            # Kill any PowerShell processes that might be running TTS
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if proc.info['name'].lower() in ['powershell.exe', 'pwsh.exe']:
                        cmdline = proc.info.get('cmdline', [])
                        if any('Speech' in str(cmd) for cmd in cmdline if cmd):
                            proc.kill()
                            print(f"Killed PowerShell TTS process: {proc.info['pid']}")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except Exception as e:
            print(f"Error killing Windows SAPI processes: {e}")

    def _cleanup_process(self):
        """Clean up any lingering processes"""
        with self.process_lock:
            if self.current_process:
                try:
                    self.current_process.terminate()
                    self.current_process.wait(timeout=1)
                except:
                    try:
                        self.current_process.kill()
                        self.current_process.wait()
                    except:
                        pass
                finally:
                    self.current_process = None

    def speak(self, text):
        """Add text to speech queue"""
        if self.running and text.strip():
            # Clear any pending items if queue is getting too full
            if self.queue.qsize() > 5:
                print("Queue full, clearing old items...")
                self.clear_queue()
            
            self.queue.put(text)
            print(f"Added to speech queue (size: {self.queue.qsize()}): '{text[:50]}...'")

    def clear_queue(self):
        """Clear all pending speech items and stop current speech"""
        # Request stop for current speech
        self.stop_requested = True
        
        # Terminate current process immediately
        self._cleanup_process()
        
        # Clear queue
        try:
            while True:
                self.queue.get_nowait()
                self.queue.task_done()
        except queue.Empty:
            pass
        
        # Reset speaking flag
        self.speaking = False
        print("Speech queue cleared and current speech stopped")

    def is_speaking(self):
        """Check if currently speaking"""
        return self.speaking and not self.stop_requested

    def stop(self):
        """Stop the speech manager"""
        if self.running:
            self.running = False
            self.stop_requested = True
            self._cleanup_process()
            self.clear_queue()
            self.queue.put(None)
            if self.worker_thread and self.worker_thread.is_alive():
                self.worker_thread.join(timeout=3)
            print("Speech Manager stopped.")


class FallbackSpeechManager:
    """
    Fallback speech manager using web browser for TTS
    """
    def __init__(self):
        self.queue = queue.Queue()
        self.running = False
        self.speaking = False
        self.worker_thread = None
        self.stop_requested = False
        
    def start(self):
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        print("Fallback Speech Manager started (using browser TTS).")
    
    def _worker(self):
        """Worker thread for browser TTS"""
        while self.running:
            try:
                text = self.queue.get(timeout=1.0)
                if text is None:
                    break
                
                # Reset stop flag for new speech
                self.stop_requested = False
                
                # Wait if currently speaking
                while self.speaking and self.running and not self.stop_requested:
                    time.sleep(0.1)
                
                if self.running and not self.stop_requested:
                    self.speaking = True
                    self._speak_browser(text)
                    # Wait a bit for the speech to complete (unless stopped)
                    speech_duration = len(text) * 0.1
                    for _ in range(int(speech_duration * 10)): # Check every 0.1 seconds
                        if self.stop_requested:
                            self._stop_browser_speech()
                            break
                        time.sleep(0.1)
                    self.speaking = False
                else:
                    print(f"Browser speech cancelled: '{text[:50]}...'")
                
                self.queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Browser TTS Error: {e}")
                self.speaking = False
                try:
                    self.queue.task_done()
                except:
                    pass
    
    def _speak_browser(self, text):
        """Use browser TTS"""
        try:
            clean_text = text.replace('"', "'").replace('\n', ' ').strip()
            if not clean_text or self.stop_requested:
                return
                
            speech_html = f"""
            <script>
                if ('speechSynthesis' in window) {{
                    // Cancel any ongoing speech first
                    speechSynthesis.cancel();
                    
                    const utterance = new SpeechSynthesisUtterance("{clean_text}");
                    utterance.rate = 0.8;
                    utterance.pitch = 1;
                    utterance.volume = 1;
                    
                    // Store reference for potential cancellation
                    window.currentUtterance = utterance;
                    
                    utterance.onstart = function() {{
                        console.log('Speech started: {clean_text[:30]}...');
                    }};
                    utterance.onend = function() {{
                        console.log('Speech ended: {clean_text[:30]}...');
                        window.currentUtterance = null;
                    }};
                    utterance.onerror = function(event) {{
                        console.error('Speech error:', event.error);
                        window.currentUtterance = null;
                    }};
                    
                    speechSynthesis.speak(utterance);
                }} else {{
                    console.error('Speech synthesis not supported');
                }}
            </script>
            """
            st.components.v1.html(speech_html, height=0)
            print(f"Browser TTS: '{clean_text[:50]}...'")
        except Exception as e:
            print(f"Browser TTS Error: {e}")

    def _stop_browser_speech(self):
        """Stop browser speech synthesis"""
        stop_html = """
        <script>
            if ('speechSynthesis' in window) {
                speechSynthesis.cancel();
                if (window.currentUtterance) {
                    window.currentUtterance = null;
                }
                console.log('Speech synthesis cancelled');
            }
        </script>
        """
        st.components.v1.html(stop_html, height=0)
        
    def speak(self, text):
        """Add text to speech queue"""
        if self.running and text.strip():
            # Clear any pending items if queue is getting too full
            if self.queue.qsize() > 3:
                self.clear_queue()
            
            self.queue.put(text)
            print(f"Added to browser TTS queue: '{text[:50]}...'")

    def clear_queue(self):
        """Clear all pending speech items and stop current speech"""
        # Request stop for current speech
        self.stop_requested = True
        
        # Stop browser speech
        self._stop_browser_speech()
        
        # Clear queue
        try:
            while True:
                self.queue.get_nowait()
                self.queue.task_done()
        except queue.Empty:
            pass
        
        # Reset speaking flag
        self.speaking = False
        print("Browser TTS queue cleared and current speech stopped")

    def is_speaking(self):
        """Check if currently speaking"""
        return self.speaking and not self.stop_requested
                
    def stop(self):
        self.running = False
        self.stop_requested = True
        self.clear_queue()
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2)
        print("Fallback Speech Manager stopped.")


def get_speech_manager():
    """Get appropriate speech manager based on environment"""
    try:
        # Try system-based TTS first
        manager = StreamlitCompatibleSpeechManager()
        manager.start()
        return manager
    except Exception as e:
        print(f"System TTS failed, using fallback: {e}")
        # Fall back to browser-based TTS
        manager = FallbackSpeechManager()
        manager.start()
        return manager


def speak_text_async(text, clear_queue=False):
    """Speak text using the appropriate speech manager"""
    if 'speech_manager' in st.session_state and text.strip():
        # Clear queue if requested (useful for interrupting previous speech)
        if clear_queue:
            st.session_state.speech_manager.clear_queue()
        
        st.session_state.speech_manager.speak(text)


def stop_current_speech():
    """Stop current speech and clear queue"""
    if 'speech_manager' in st.session_state:
        st.session_state.speech_manager.clear_queue()


def is_currently_speaking():
    """Check if TTS is currently active"""
    if 'speech_manager' in st.session_state:
        return st.session_state.speech_manager.is_speaking()
    return False


# Set page config
st.set_page_config(
    page_title="NÜMIRROR - AI Fashion & Emotion Assistant",
    page_icon="🪞",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "NÜMIRROR - Your Personal AI Fashion & Emotion Assistant"
    }
)

# Initialize session state
if 'current_mode' not in st.session_state:
    st.session_state.current_mode = None
if 'last_outfit_description' not in st.session_state:
    st.session_state.last_outfit_description = ""
if 'last_emotion' not in st.session_state:
    st.session_state.last_emotion = ""
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'initial_message_sent' not in st.session_state:
    st.session_state.initial_message_sent = False
if 'current_emotion_detected' not in st.session_state:
    st.session_state.current_emotion_detected = ""
if 'emotion_confidence' not in st.session_state:
    st.session_state.emotion_confidence = 0.0

# Initialize speech manager
if 'speech_manager' not in st.session_state:
    st.session_state.speech_manager = get_speech_manager()
    atexit.register(lambda: st.session_state.speech_manager.stop())


@st.cache_resource
def load_fashion_model():
    """Load and cache the fashion detection model"""
    try:
        fashion_model_id = "valentinafeve/yolos-fashionpedia"
        fashion_processor = YolosFeatureExtractor.from_pretrained(fashion_model_id, use_fast=False)
        fashion_model = YolosForObjectDetection.from_pretrained(fashion_model_id)
        fashion_model = fashion_model.to("cuda" if torch.cuda.is_available() else "cpu")
        return fashion_processor, fashion_model
    except Exception as e:
        st.error(f"Error loading fashion model: {e}")
        return None, None

@st.cache_resource
def load_emotion_model():
    """Load and cache the FER emotion detection model"""
    try:
        # Initialize FER with MTCNN for better face detection
        emotion_detector = FER(mtcnn=True)
        print("FER emotion detector loaded successfully")
        return emotion_detector
    except Exception as e:
        st.error(f"Error loading FER emotion model: {e}")
        return None

@st.cache_resource
def load_nlp_model():
    """Load and cache the NLP model"""
    try:
        nlp_model_path = r"C:\Users\yashu\OneDrive\Desktop\yashu\final1\cpu_and_mobile\cpu-int4-rtn-block-32-acc-level-4"
        if not os.path.exists(nlp_model_path):
            st.error(f"NLP model not found at {nlp_model_path}")
            return None, None
        
        nlp_model = Model(nlp_model_path)
        nlp_tokenizer = Tokenizer(nlp_model)
        return nlp_model, nlp_tokenizer
    except Exception as e:
        st.error(f"Error loading NLP model: {e}")
        return None, None

def initialize_speech_recognition():
    """Initialize speech recognition"""
    try:
        recognizer = sr.Recognizer()
        return recognizer
    except Exception as e:
        st.error(f"Error initializing speech recognition: {e}")
        return None

def detect_clothing_type(image_bgr, fashion_processor, fashion_model, threshold=0.8):
    """Detect clothing items in the image"""
    INCLUDE_LABELS = {
        'shirt, blouse', 'top, t-shirt, sweatshirt', 'sweater', 'jacket', 'pants',
        'shorts', 'skirt', 'coat', 'dress', 'jumpsuit', 'shoe', 'sleeve'
    }
    
    try:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        inputs = fashion_processor(images=pil_image, return_tensors="pt").to(fashion_model.device)
        
        with torch.no_grad():
            outputs = fashion_model(**inputs)
        
        results = fashion_processor.post_process_object_detection(
            outputs,
            target_sizes=[(pil_image.height, pil_image.width)],
            threshold=0.4
        )[0]

        labels = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            confidence = score.item()
            label_name = fashion_model.config.id2label[label.item()]
            if confidence < threshold or label_name not in INCLUDE_LABELS:
                continue

            box = [round(i, 2) for i in box.tolist()]
            x0, y0, x1, y1 = map(int, box)
            crop = image_rgb[y0:y1, x0:x1]

            if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                dominant_color = None
            else:
                crop_pil = Image.fromarray(crop)
                buffer = io.BytesIO()
                crop_pil.save(buffer, format="JPEG")
                buffer.seek(0)
                try:
                    color_thief = ColorThief(buffer)
                    dominant_color = color_thief.get_color(quality=1)
                    color_name = color_rgb_to_name(dominant_color)
                except Exception as e:
                    dominant_color = None
            display_label = "shirt/t-shirt" if label_name == "sleeve" else label_name
            labels.append({
                "label": display_label,
                "score": round(confidence, 2),
                "box": box,
                "color": dominant_color,
            })
        return labels
    except Exception as e:
        st.error(f"Error in clothing detection: {e}")
        return []

def detect_emotion(image_array, emotion_detector):
    """
    Detect emotion from the image using FER
    Returns the top emotion and confidence score
    """
    try:
        # Convert image to RGB if it's BGR (from OpenCV)
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            # Assume it's BGR from camera input, convert to RGB
            image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image_array
        
        # Get emotion predictions for all faces detected
        emotion_predictions = emotion_detector.detect_emotions(image_rgb)
        
        if not emotion_predictions:
            return "No face detected", 0.0
        
        # Get the first (usually largest) face's emotions
        emotions = emotion_predictions[0]["emotions"]
        
        # Find the emotion with highest confidence
        top_emotion = max(emotions, key=emotions.get)
        confidence = emotions[top_emotion]
        
        return top_emotion.capitalize(), round(confidence, 3)
        
    except Exception as e:
        st.error(f"Error in FER emotion detection: {e}")
        return "Error", 0.0

def generate_response(prompt,sys_tokens, nlp_model, nlp_tokenizer):
    """Generate AI response"""
    try:
        token = nlp_tokenizer.encode(prompt)
        tokens = np.array(sys_tokens + token, dtype=np.int32)
        params = GeneratorParams(nlp_model)
        params.input_ids = np.array(tokens, dtype=np.int32)
        params.set_search_options(
            max_length=800,
            temperature=0.7,
            top_p=0.9,
            early_stopping=False,
            length_penalty=1.0
        )

        generator = Generator(nlp_model, params)
        streamer = nlp_tokenizer.create_stream()
        full_response = ""
        while not generator.is_done():
            generator.compute_logits()
            generator.generate_next_token()
            tok = generator.get_next_tokens()[0]
            text = streamer.decode(tok)
            if text:
                full_response += text

        cleaned = full_response.strip()
        if cleaned.lower().startswith("ai:"):
            cleaned = cleaned[len("ai:"):].strip()
        return cleaned
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "I'm sorry, I'm having trouble generating a response right now."

def draw_detections(image_bgr, detections):
    """Draw detection boxes on image"""
    drawn_labels = set()
    for item in detections:
        label = item["label"]
        score = item["score"]
        box = item["box"]
        color = item.get("color", (0, 255, 0))
        color_bgr = (color[2], color[1], color[0]) if color else (0, 255, 0)
        color_name = item.get("color_name", "Unknown")

        if label in drawn_labels:
            continue
        drawn_labels.add(label)

        x0, y0, x1, y1 = map(int, box)
        cv2.rectangle(image_bgr, (x0, y0), (x1, y1), color_bgr, 2)
        text = f"{label}: {score*100:.1f}% ({color_name})"
        cv2.putText(image_bgr, text, (x0, y0 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)
    return image_bgr

def draw_emotion_on_image(image_array, emotion_detector):
    """Draw emotion detection results on image"""
    try:
        # Convert to RGB for FER processing
        if len(image_array.shape) == 3:
            image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image_array
            
        # Detect emotions
        emotion_predictions = emotion_detector.detect_emotions(image_rgb)
        
        # Convert back to BGR for OpenCV drawing
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        for face_data in emotion_predictions:
            # Get bounding box
            box = face_data["box"]
            x, y, w, h = box
            
            # Get top emotion
            emotions = face_data["emotions"]
            top_emotion = max(emotions, key=emotions.get)
            confidence = emotions[top_emotion]
            
            # Draw bounding box
            cv2.rectangle(image_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw emotion text
            text = f"{top_emotion.capitalize()}: {confidence:.2f}"
            cv2.putText(image_bgr, text, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
        return image_bgr
        
    except Exception as e:
        st.error(f"Error drawing emotion on image: {e}")
        return image_array

def color_rgb_to_name(rgb_color):
    """Convert RGB color to a readable color name with more colors including pink"""
    if not rgb_color:
        return "unknown"
    
    r, g, b = rgb_color
    
    if r > 240 and g > 240 and b > 240:
        return "white"
    elif r < 50 and g < 50 and b < 50:
        return "black"
    elif r > 200 and g < 80 and b < 80:
        return "red"
    elif r < 80 and g > 200 and b < 80:
        return "green"
    elif r < 80 and g < 80 and b > 200:
        return "blue"
    elif r > 200 and g > 200 and b < 80:
        return "yellow"
    elif r > 200 and g < 80 and b > 200:
        return "magenta"
    elif r < 80 and g > 200 and b > 200:
        return "cyan"
    elif r > 200 and g > 100 and b < 80:
        return "orange"
    elif r > 100 and g > 100 and b > 100:
        return "gray"
    elif r > 120 and g > 80 and b < 80:
        return "brown"
    elif r > 150 and g > 50 and b > 150:
        return "purple"
    elif r > 180 and g > 100 and b > 180:
        return "lavender"
    elif r > 170 and g > 120 and b > 180:
        return "violet"
    elif r > 120 and g < 50 and b < 20:
        return "maroon"
    elif r > 50 and g > 120 and b < 20:
        return "olive"
    elif r < 30 and g > 180 and b > 80:
        return "spring green"
    elif r > 200 and 100 < g < 180 and 150 < b < 220:
        return "pink"
    else:
        return "mixed-color"



def listen_for_speech_streamlit(recognizer):
    """Listen for speech input with Streamlit integration"""
    try:
        with sr.Microphone() as source:
            st.info("🎤 Listening... Please speak now!")
            recognizer.pause_threshold = 1
            recognizer.adjust_for_ambient_noise(source, duration=1)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                audio = recognizer.listen(source, timeout=10, phrase_time_limit=10)
                progress_bar.progress(100)
                status_text.text("🔄 Processing speech...")
                
            except sr.WaitTimeoutError:
                progress_bar.empty()
                status_text.empty()
                st.warning("⏰ No speech detected. Please try again.")
                return ""
            
        try:
            query = recognizer.recognize_google(audio, language='en-in')
            progress_bar.empty()
            status_text.empty()
            st.success(f"👤 You said: {query}")
            return query.lower()
            
        except sr.UnknownValueError:
            progress_bar.empty()
            status_text.empty()
            st.warning("🤔 Sorry, I couldn't understand what you said. Please try again.")
            return ""
        except sr.RequestError as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Speech recognition error: {e}")
            return ""
            
    except Exception as e:
        st.error(f"❌ Microphone error: {e}")
        return ""

nlp_model, nlp_tokenizer = load_nlp_model()
def main():
    st.markdown("""
    <style>
    [data-testid="stSidebar"] {
    display: block !important;
    visibility: visible !important;
    }

    [data-testid="stSidebar"][aria-expanded="false"] {
    display: block !important;
    margin-left: 0 !important;
    }

/* Make collapse/expand button ALWAYS visible and prominent */
    [data-testid="collapsedControl"] {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
    z-index: 999999 !important;
    position: fixed !important;
    left: 0 !important;
    top: 50% !important;
    background: #8b5cf6 !important;
    color: #ffffff !important;
    border: 3px solid #ffffff !important;
    border-radius: 0 12px 12px 0 !important;
    padding: 16px 8px !important;
    width: 40px !important;
    height: 60px !important;
    cursor: pointer !important;
    box-shadow: 0 4px 15px rgba(139, 92, 246, 0.8) !important;
    transition: all 0.3s ease !important;
    }

    [data-testid="collapsedControl"]:hover {
    background: #7c3aed !important;
    box-shadow: 0 6px 25px rgba(139, 92, 246, 1) !important;
    transform: translateX(5px) !important;
    width: 50px !important;
    }

    [data-testid="collapsedControl"] svg {
    fill: #ffffff !important;
    width: 28px !important;
    height: 28px !important;
    }

/* Ensure sidebar close button (X) is visible when sidebar is open */
   [data-testid="stSidebar"] button[kind="header"],
   [data-testid="stSidebar"] [data-testid="baseButton-header"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    color: #ffffff !important;
    background: rgba(139, 92, 246, 0.3) !important;
    border-radius: 8px !important;
    padding: 8px !important;
    margin: 8px !important;
    }

    [data-testid="stSidebar"] button[kind="header"]:hover {
    background: rgba(139, 92, 246, 0.5) !important;
    }

    [data-testid="stSidebar"] button[kind="header"] svg {
    fill: #ffffff !important;
    width: 24px !important;
    height: 24px !important;
    }

    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main App Background - Deep Black */
    .stApp {
        background: #0a0a0a;
        color: #ffffff;
    }
    
    /* Main Title - Large, Bold, White */
    .main-title {
        font-size: 72px;
        font-weight: 800;
        text-align: center;
        color: #ffffff;
        padding: 30px 0 10px 0;
        letter-spacing: 4px;
        text-shadow: 0 0 30px rgba(139, 92, 246, 0.5);
        animation: glow 3s ease-in-out infinite;
    }
    
    @keyframes glow {
        0%, 100% {
            text-shadow: 0 0 30px rgba(139, 92, 246, 0.5);
        }
        50% {
            text-shadow: 0 0 50px rgba(139, 92, 246, 0.8);
        }
    }
    
    /* Subtitle - Off-White/Light Gray */
    .subtitle {
        text-align: center;
        color: #d1d5db;
        font-size: 20px;
        margin-bottom: 30px;
        font-weight: 400;
        letter-spacing: 1px;
    }
    
    /* Sidebar - Dark with Subtle Gradient */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f0f 0%, #000000 100%);
        border-right: 2px solid #1f1f1f;
    }
    
    [data-testid="stSidebar"] h1 {
        color: #ffffff !important;
        font-size: 24px !important;
        text-align: center;
        padding: 10px 0;
    }
    
    [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
        font-size: 18px !important;
        margin-top: 20px;
    }
    
    [data-testid="stSidebar"] p {
        color: #e5e7eb;
    }
    
    /* Sidebar Buttons - Fixed White Text Issue */
    [data-testid="stSidebar"] .stButton>button {
        background: linear-gradient(135deg, #1f1f1f 0%, #2a2a2a 100%);
        color: #ffffff !important;
        border: 2px solid #3f3f3f;
        border-radius: 12px;
        font-weight: 600;
        font-size: 16px;
        padding: 16px 20px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.5);
        width: 100%;
        margin: 8px 0;
    }
    
    [data-testid="stSidebar"] .stButton>button:hover {
        background: linear-gradient(135deg, #2a2a2a 0%, #3f3f3f 100%);
        border-color: #8b5cf6;
        box-shadow: 0 0 20px rgba(139, 92, 246, 0.6);
        transform: translateY(-2px);
    }
    
    /* Fashion Mode Button Accent */
    [data-testid="stSidebar"] button[key="fashion_btn"] {
        border-left: 4px solid #ef4444;
    }
    
    [data-testid="stSidebar"] button[key="fashion_btn"]:hover {
        border-color: #ef4444;
        box-shadow: 0 0 20px rgba(239, 68, 68, 0.6);
    }
    
    /* Emotion Mode Button Accent */
    [data-testid="stSidebar"] button[key="emotion_btn"] {
        border-left: 4px solid #10b981;
    }
    
    [data-testid="stSidebar"] button[key="emotion_btn"]:hover {
        border-color: #10b981;
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.6);
    }
    
    /* General Mode Button Accent */
    [data-testid="stSidebar"] button[key="general_btn"] {
        border-left: 4px solid #8b5cf6;
    }
    
    [data-testid="stSidebar"] button[key="general_btn"]:hover {
        border-color: #8b5cf6;
        box-shadow: 0 0 20px rgba(139, 92, 246, 0.6);
    }
    
    /* Reset Button Accent */
    [data-testid="stSidebar"] button[key="reset_btn"] {
        border-left: 4px solid #f59e0b;
        background: linear-gradient(135deg, #2a1810 0%, #1f1410 100%);
    }
    
    [data-testid="stSidebar"] button[key="reset_btn"]:hover {
        border-color: #f59e0b;
        box-shadow: 0 0 20px rgba(245, 158, 11, 0.6);
    }
    
    /* Main Content Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #1f1f1f 0%, #2a2a2a 100%);
        color: #ffffff !important;
        border: 2px solid #3f3f3f;
        border-radius: 12px;
        font-weight: 600;
        font-size: 15px;
        padding: 12px 24px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.5);
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #2a2a2a 0%, #3f3f3f 100%);
        border-color: #8b5cf6;
        box-shadow: 0 0 15px rgba(139, 92, 246, 0.6);
        transform: translateY(-2px);
    }
    
    /* Card Containers */
    [data-testid="column"] {
        background: #0f0f0f;
        border: 2px solid #1f1f1f;
        border-radius: 16px;
        padding: 24px;
        margin: 10px 0;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.6);
    }
    
    /* Section Headers - White & Bold */
    .stApp h2 {
        color: #ffffff !important;
        font-weight: 700;
        font-size: 28px;
        margin-bottom: 16px;
        padding-bottom: 10px;
        border-bottom: 2px solid #8b5cf6;
    }
    
    .stApp h3 {
        color: #ffffff !important;
        font-weight: 600;
        font-size: 22px;
        margin-top: 20px;
    }
    
    /* Info Boxes - Dark Theme */
    .stInfo {
        background: rgba(139, 92, 246, 0.15) !important;
        border-left: 4px solid #8b5cf6 !important;
        border-radius: 12px;
        color: #e5e7eb !important;
        padding: 16px;
        animation: fadeIn 0.5s ease;
    }
    
    .stSuccess {
        background: rgba(16, 185, 129, 0.15) !important;
        border-left: 4px solid #10b981 !important;
        border-radius: 12px;
        color: #e5e7eb !important;
        padding: 16px;
    }
    
    .stWarning {
        background: rgba(245, 158, 11, 0.15) !important;
        border-left: 4px solid #f59e0b !important;
        border-radius: 12px;
        color: #e5e7eb !important;
        padding: 16px;
    }
    
    .stError {
        background: rgba(239, 68, 68, 0.15) !important;
        border-left: 4px solid #ef4444 !important;
        border-radius: 12px;
        color: #e5e7eb !important;
        padding: 16px;
    }
    
    /* Camera Feed Styling */
    [data-testid="stImage"] {
        border-radius: 16px;
        border: 3px solid #1f1f1f;
        overflow: hidden;
        box-shadow: 0 8px 25px rgba(139, 92, 246, 0.3);
        transition: all 0.3s ease;
    }
    
    [data-testid="stImage"]:hover {
        box-shadow: 0 8px 35px rgba(139, 92, 246, 0.5);
        border-color: #8b5cf6;
    }
    
    [data-testid="stCameraInput"] {
        background: #0f0f0f;
        border-radius: 16px;
        padding: 20px;
        border: 2px solid #1f1f1f;
    }
    
    [data-testid="stCameraInput"] button {
        background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px;
        padding: 14px 28px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(139, 92, 246, 0.4);
    }
    
    [data-testid="stCameraInput"] button:hover {
        background: linear-gradient(135deg, #7c3aed 0%, #6d28d9 100%) !important;
        box-shadow: 0 6px 25px rgba(139, 92, 246, 0.6);
        transform: scale(1.05);
    }
    
    /* Mode Card - Active Mode Display */
    .mode-card {
        background: linear-gradient(135deg, #1f1f1f 0%, #0f0f0f 100%);
        border: 2px solid #8b5cf6;
        padding: 24px;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 0 30px rgba(139, 92, 246, 0.4);
        animation: pulse 2s infinite;
    }
    
    .mode-card h2 {
        margin: 0;
        font-size: 32px;
        font-weight: 700;
        color: #ffffff !important;
        border: none !important;
    }
    
    .mode-card p {
        margin: 8px 0 0 0;
        opacity: 0.9;
        font-size: 16px;
        color: #d1d5db;
    }
    
    /* Chat Messages */
    .chat-user {
        background: linear-gradient(135deg, #1e3a5f 0%, #1a2f4d 100%);
        padding: 16px 20px;
        border-radius: 18px 18px 5px 18px;
        margin: 12px 0;
        border-left: 4px solid #3b82f6;
        color: #ffffff;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
        animation: slideInLeft 0.3s ease;
    }
    
    .chat-ai {
        background: linear-gradient(135deg, #2d1b4e 0%, #1f1435 100%);
        padding: 16px 20px;
        border-radius: 18px 18px 18px 5px;
        margin: 12px 0;
        border-left: 4px solid #8b5cf6;
        color: #ffffff;
        box-shadow: 0 4px 12px rgba(139, 92, 246, 0.3);
        animation: slideInRight 0.3s ease;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes pulse {
        0%, 100% {
            box-shadow: 0 0 30px rgba(139, 92, 246, 0.4);
        }
        50% {
            box-shadow: 0 0 45px rgba(139, 92, 246, 0.7);
        }
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background: #1f1f1f !important;
        border-radius: 12px;
        border: 2px solid #2a2a2a;
        color: #ffffff !important;
        font-weight: 600;
        padding: 12px;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: #2a2a2a !important;
        border-color: #8b5cf6;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 12px;
        height: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0a0a0a;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
        border-radius: 10px;
        border: 2px solid #0a0a0a;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #7c3aed 0%, #6d28d9 100%);
    }
    
    /* Text & Paragraphs - White/Light Gray */
    p, li, span, div {
        color: #e5e7eb !important;
    }
    
    /* Markdown Text */
    [data-testid="stMarkdownContainer"] {
        color: #e5e7eb;
    }
    
    /* Dividers */
    hr {
        border-color: #1f1f1f !important;
        margin: 24px 0;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-color: #8b5cf6 transparent transparent transparent !important;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Captions */
    .stCaption {
        color: #9ca3af !important;
        font-size: 14px;
    }
    
    /* Toggle/Checkbox Styling */
    [data-testid="stCheckbox"] label {
        color: #ffffff !important;
    }
    
    /* Active Models Section in Sidebar */
    [data-testid="stSidebar"] .stInfo {
        background: rgba(16, 185, 129, 0.2) !important;
        border: 2px solid #10b981;
        color: #ffffff !important;
    }
    </style>
    """, unsafe_allow_html=True)  
    st.markdown('<h1 class="main-title">🪞 NÜMIRROR</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Your Personal AI Fashion & Emotion Assistant</p>', unsafe_allow_html=True)
    st.markdown("---")

    # Load models
    if not st.session_state.models_loaded:
        with st.spinner("Loading AI models... This may take a moment."):
            fashion_processor, fashion_model = load_fashion_model()
            emotion_detector = load_emotion_model() # Changed from emotion_model to emotion_detector
            # nlp_model, nlp_tokenizer = load_nlp_model()
            speech_recognizer = initialize_speech_recognition()
            
            if all([fashion_processor, fashion_model, emotion_detector, nlp_model, nlp_tokenizer, speech_recognizer]):
                st.session_state.fashion_processor = fashion_processor
                st.session_state.fashion_model = fashion_model
                st.session_state.emotion_detector = emotion_detector # Store FER detector
                st.session_state.nlp_model = nlp_model
                st.session_state.nlp_tokenizer = nlp_tokenizer
                st.session_state.speech_recognizer = speech_recognizer
                st.session_state.models_loaded = True
                st.success("All models loaded successfully! FER emotion detection ready.")
                speak_text_async("All models are ready. FER emotion detection is active. Please select a mode.")
            else:
                st.error("Failed to load some models. Please check your model paths and dependencies.")
                return

    # Sidebar for mode selection
    st.sidebar.title("🎯 Select Mode")
    st.sidebar.markdown("---")
    
    if st.sidebar.button("👗 \nFashion Mode", use_container_width=True, key="fashion_btn"):
            stop_current_speech() # Stop any ongoing speech
            st.session_state.current_mode = "fashion"
            st.session_state.conversation_history = []
            st.session_state.initial_message_sent = False
            st.session_state.current_emotion_detected = ""
            st.session_state.emotion_confidence = 0.0
            
    if st.sidebar.button("🎭 \nEmotion Mode", use_container_width=True, key="emotion_btn"):
            stop_current_speech() # Stop any ongoing speech
            st.session_state.current_mode = "emotion"
            st.session_state.conversation_history = []
            st.session_state.initial_message_sent = False
            st.session_state.current_emotion_detected = ""
            st.session_state.emotion_confidence = 0.0
    
    if st.sidebar.button("💬 \nGeneral", use_container_width=True, key="general_btn"):
            stop_current_speech()
            st.session_state.current_mode = "general"
            st.session_state.conversation_history = []
            st.session_state.initial_message_sent = False
            st.session_state.current_emotion_detected = ""
            st.session_state.emotion_confidence = 0.0
            welcome_msg = "General mode activated. I'm ready to answer any questions you have!"
            st.session_state.conversation_history.append({
               "role": "assistant",
               "content": welcome_msg
            })
            st.session_state.initial_message_sent = True
    st.sidebar.markdown("---")
    # Reset button
    if st.sidebar.button("🔄 Reset", use_container_width=True, key="reset_btn"):
        stop_current_speech() # Stop any ongoing speech
        st.session_state.current_mode = None
        st.session_state.conversation_history = []
        st.session_state.last_outfit_description = ""
        st.session_state.last_emotion = ""
        st.session_state.initial_message_sent = False
        st.session_state.current_emotion_detected = ""
        st.session_state.emotion_confidence = 0.0
        st.rerun()

    # Display current model info in sidebar
    if st.session_state.models_loaded:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🤖 Active Models")
        st.sidebar.info("✅ FER Emotion Detection\n✅ YOLOS Fashion Detection\n✅ Speech Recognition\n✅ Text-to-Speech")

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📷 Camera Feed")
        
        if st.session_state.current_mode == "general":
           st.info("📝 General Mode - No camera needed! Just ask me anything using voice commands.")
           camera_input = None
        else:
           camera_input = st.camera_input("Take a picture",key="main_camera")
        if camera_input is not None and st.session_state.current_mode:
            image = Image.open(camera_input)
            image_array = np.array(image)
            
            if st.session_state.current_mode == "fashion":
                st.write("🔍 Analyzing your outfit...")
                
                detections = detect_clothing_type(
                    image_array, 
                    st.session_state.fashion_processor, 
                    st.session_state.fashion_model
                )
                
                if detections:
                    image_with_boxes = draw_detections(image_array.copy(), detections)
                    st.image(image_with_boxes, caption="Detected Clothing Items", use_container_width=True)
                    
                    clothing_list = [
                        f"a {color_rgb_to_name(item.get('color'))} {item['label']}"
                        for item in detections
                    ]
                    st.session_state.last_outfit_description = ", ".join(clothing_list)
                    
                    detection_text = f"I can see you're wearing {st.session_state.last_outfit_description}. What would you like to know about your outfit?"
                    
                    if not st.session_state.initial_message_sent:
                        st.session_state.conversation_history.append({
                            "role": "assistant",
                            "content": detection_text
                        })
                        st.session_state.initial_message_sent = True
                        speak_text_async(detection_text)
                    
                    st.write("*Detected Items:*")
                    for item in detections:
                        color_info = f"Color: {color_rgb_to_name(item.get('color'))}" if item.get('color') else "Color: Unknown"
                        st.write(f"- {item['label']} ({item['score']*100:.1f}% confidence) - {color_info}")
                        
                else:
                    st.write("❌ No clothing items detected. Please try again with better lighting.")
                    no_detection_text = "I couldn't detect any specific clothing items. Try taking another photo with better lighting."
                    
                    if not st.session_state.initial_message_sent:
                        st.session_state.conversation_history.append({
                            "role": "assistant", 
                            "content": no_detection_text
                        })
                        st.session_state.initial_message_sent = True
                        speak_text_async(no_detection_text)

            elif st.session_state.current_mode == "emotion":
                st.write("😊 Analyzing your emotion with FER...")
                
                # Use FER for emotion detection
                emotion, confidence = detect_emotion(image_array, st.session_state.emotion_detector)
                st.session_state.last_emotion = emotion
                st.session_state.emotion_confidence = confidence
                
                # Draw emotion detection results on image
                image_with_emotion = draw_emotion_on_image(image_array, st.session_state.emotion_detector)
                st.image(image_with_emotion, caption=f"Detected Emotion: {emotion}", use_container_width=True)
                
                if emotion != "No face detected" and emotion != "Error":
                    st.write(f"*Detected Emotion:* {emotion}")
                    st.write(f"*Confidence:* {confidence:.1%}")
                    
                    # Create more detailed emotion feedback
                    if confidence >= 0.6:
                        confidence_text = "high confidence"
                    elif confidence >= 0.4:
                        confidence_text = "moderate confidence"
                    else:
                        confidence_text = "low confidence"
                    
                    emotion_text = f"I can see you're feeling {emotion.lower()} with {confidence_text}. How can I help support your mood today?"
                else:
                    emotion_text = f"{emotion}. Please make sure your face is clearly visible in the image."
                
                if not st.session_state.initial_message_sent or st.session_state.current_emotion_detected != emotion:
                    st.session_state.conversation_history.append({
                        "role": "assistant",
                        "content": emotion_text
                    })
                    st.session_state.initial_message_sent = True
                    st.session_state.current_emotion_detected = emotion
                    speak_text_async(emotion_text)

    with col2:
        st.subheader("💬 Conversation")
        if st.session_state.current_mode:
            mode_icons = {"fashion": "👗", "emotion": "🎭", "general": "💬"}
            mode_names = {"fashion": "Fashion Mode", "emotion": "Emotion Mode", "general": "General Mode"}
    
            st.markdown(f"""
                <div class='mode-card'>
                  <h2 style='margin:0; font-size: 28px;'>{mode_icons[st.session_state.current_mode]} {mode_names[st.session_state.current_mode]}</h2>
                  <p style='margin:5px 0 0 0; opacity: 0.9; font-size: 16px;'>Active & Ready</p>
                </div>
             """, unsafe_allow_html=True)
            # --- MODIFICATION START: ADD THE CONTEXT TOGGLE ---
            # Only show the toggle in fashion mode and only after an outfit has been detected.
            # This keeps the UI clean.
            if st.session_state.current_mode == "fashion" and st.session_state.last_outfit_description:
                st.toggle(
                    label="Ask about my current outfit",
                    value=True, # Default to ON after a picture is taken
                    key="ask_about_outfit",
                    help="Turn this ON to ask questions about the detected outfit. Turn it OFF for general fashion advice."
                )
            # --- MODIFICATION END ---

        else:
            st.info("Please select a mode from the sidebar to get started!")

        if st.session_state.current_mode and (st.session_state.last_outfit_description or st.session_state.last_emotion or st.session_state.current_mode == "general"):
            
            chat_container = st.container()
            with chat_container:
                for message in st.session_state.conversation_history:
                    if message["role"] == "user":
                        st.write(f"👤 *You:* {message['content']}")
                    else:
                        st.write(f"🤖 *AI:* {message['content']}")
            
            st.markdown("### 🎤 Voice Commands")
            st.info("Click the button below and speak your question or command!")
            
            col1_btn, col2_btn, col3_btn = st.columns(3)
            with col1_btn:
                listen_button = st.button("🎤 Listen", use_container_width=True)
            with col2_btn:
                speak_button = st.button("🔊 Repeat Last", use_container_width=True)
            with col3_btn:
                stop_button = st.button("🛑 Stop Speech", use_container_width=True)
            
            if listen_button:
                user_speech = listen_for_speech_streamlit(st.session_state.speech_recognizer)
                
                if user_speech:
                    st.session_state.conversation_history.append({
                        "role": "user",
                        "content": user_speech
                    })
                    
                    if any(word in user_speech for word in ["quit", "exit", "bye", "goodbye"]):
                        farewell_msg = "Goodbye! Thank you for using the Fashion & Mood Assistant!"
                        st.session_state.conversation_history.append({
                            "role": "assistant",
                            "content": farewell_msg
                        })
                        speak_text_async(farewell_msg, clear_queue=True)
                        st.balloons()
                        
                    elif any(word in user_speech for word in ["stop", "reset", "clear", "start over"]):
                        st.session_state.current_mode = None
                        st.session_state.conversation_history = []
                        st.session_state.last_outfit_description = ""
                        st.session_state.last_emotion = ""
                        st.session_state.initial_message_sent = False
                        st.session_state.current_emotion_detected = ""
                        st.session_state.emotion_confidence = 0.0
                        reset_msg = "Resetting the assistant. Please select a mode to start again."
                        speak_text_async(reset_msg, clear_queue=True)
                        st.rerun()
                        
                    else:
                        with st.spinner("🤖 Generating response..."):
                            # --- MODIFICATION START: DYNAMIC PROMPT GENERATION ---
                            if st.session_state.current_mode == "fashion":
                                # Start with a general, base prompt
                                system_prompt = (
                                    "You are a helpful and friendly fashion AI assistant. "
                                    "Provide practical fashion advice. Keep responses conversational and helpful. "
                                    "Do not provide more than 3 options and keep them in 1-2 sentences each."
                                )
                                system_prompt_tokens=nlp_tokenizer.encode(system_prompt)
                                # Check the state of our new toggle.
                                # Use .get() for safety in case the key doesn't exist yet.
                                if st.session_state.get('ask_about_outfit', False) and st.session_state.last_outfit_description:
                                    # If the toggle is ON, prepend the specific outfit context.
                                    outfit_context = (
                                        f"Context: The user is asking about their current outfit, which consists of: "
                                        f"{st.session_state.last_outfit_description}.\n"
                                    )
                                    system_prompt = outfit_context + system_prompt
                                    system_prompt_tokens=nlp_tokenizer.encode(system_prompt)
                            
                            elif st.session_state.current_mode == "emotion": # This is the original, unchanged logic for emotion mode
                                emotion_context = f"The user's detected emotion is {st.session_state.last_emotion}"
                                if st.session_state.emotion_confidence > 0:
                                    emotion_context += f" with {st.session_state.emotion_confidence:.1%} confidence"
                                
                                system_prompt = (
                                    f"You are a supportive and friendly emotional wellness assistant. {emotion_context}.\n"
                                    "Provide supportive and empathetic responses that acknowledge their emotional state. "
                                    "Offer practical suggestions for managing emotions when appropriate. "
                                    "Keep responses warm, understanding, and in 3-4 sentences. Do not mention clothing or fashion."
                                )
                                system_prompt_tokens=nlp_tokenizer.encode(system_prompt)
                            #  --- MODIFICATION END ---
                            else:
                                system_prompt = (
                                    "You are a helpful and knowledgeable AI assistant. "
                                    "Answer questions accurately and concisely across any topic. "
                                    "Keep responses conversational, informative, and limited to 3-4 sentences. "
                                    "Be friendly and helpful."
                                )
                                system_prompt_tokens=nlp_tokenizer.encode(system_prompt)
                            current_turn = f"<|user|>\n{user_speech}<|end|>\n<|assistant|>\n"
                            # full_prompt = f"<|system|>\n{system_prompt}<|end|>\n{current_turn}"
                            
                            ai_response = generate_response(current_turn,system_prompt_tokens,st.session_state.nlp_model, st.session_state.nlp_tokenizer)
                            
                            st.session_state.conversation_history.append({
                                "role": "assistant",
                                "content": ai_response
                            })
                            
                            speak_text_async(ai_response)
                    
                    st.rerun()
            
            if speak_button and st.session_state.conversation_history:
                last_ai_response = [msg for msg in st.session_state.conversation_history if msg["role"] == "assistant"]
                if last_ai_response:
                    speak_text_async(last_ai_response[-1]["content"], clear_queue=True)
                    st.success("🔊 Speaking last response...")
            
            if stop_button:
                stop_current_speech()
                st.info("🛑 Stopped current speech and cleared queue.")
            
            with st.expander("ℹ Voice Commands Help & FER Information"):
                st.markdown("""
                *Available Voice Commands:*
                - "quit", "exit", "bye", "goodbye" - End the session
                - "stop", "reset", "clear", "start over" - Reset the assistant
                - Any fashion question (in Fashion Mode) - Get style advice
                - Any emotional question (in Emotion Mode) - Get supportive responses
                
                *Tips for better speech recognition:*
                - Speak clearly and at a normal pace
                - Ensure you're in a quiet environment
                - Wait for the "Listening..." indicator before speaking
                - Keep questions concise and clear
                
                *FER Emotion Detection Features:*
                - Uses state-of-the-art Face Emotion Recognition
                - Includes MTCNN for improved face detection
                - Provides confidence scores for emotion predictions
                - Detects multiple faces if present
                - Real-time emotion analysis with bounding boxes
                
                *Supported Emotions:*
                - Happy, Sad, Angry, Fear, Surprise, Disgust, Neutral
                
                *TTS Information:*
                - System will automatically detect the best TTS method for your platform
                - On Windows: Uses built-in SAPI
                - On macOS: Uses built-in 'say' command  
                - On Linux: Uses espeak, spd-say, or festival
                - Fallback: Browser-based Web Speech API
                
                *Speech Control:*
                - Use the "Stop Speech" button to immediately halt any ongoing speech
                - Reset or mode changes will automatically stop current speech
                - Queue is automatically cleared when stopping speech
                """)
                
            # Display current detection info
            if st.session_state.current_mode == "emotion" and st.session_state.last_emotion:
                st.sidebar.markdown("---")
                st.sidebar.markdown("### 🎭 Current Detection")
                st.sidebar.write(f"*Emotion:* {st.session_state.last_emotion}")
                if st.session_state.emotion_confidence > 0:
                    st.sidebar.write(f"*Confidence:* {st.session_state.emotion_confidence:.1%}")
                    
            elif st.session_state.current_mode == "fashion" and st.session_state.last_outfit_description:
                st.sidebar.markdown("---")
                st.sidebar.markdown("### 👗 Current Outfit")
                st.sidebar.write(f"*Detected:* {st.session_state.last_outfit_description}")
                
            st.markdown("---")
            st.caption("🎙 Make sure your microphone is working and permissions are granted.")

        elif st.session_state.current_mode:
            st.write("📸 Please take a picture first to start the conversation!")
            if st.session_state.current_mode == "emotion":
                st.info("📝 *FER Emotion Detection:* Make sure your face is clearly visible and well-lit for best results.")


if __name__ == "__main__":
    main()
