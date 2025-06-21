import sys
import collections
import numpy as np
import cv2  # For camera capture
import pyvirtualcam
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QProgressBar,
    QLabel,
    QHBoxLayout,
)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
import speech_recognition as sr
import webrtcvad
import pyaudio

# Local import (ensure src is on PYTHONPATH or run as module)
from frame_handler import FrameHandler

class VoiceActivityDetector:
    """Handles voice activity detection using WebRTC VAD.
    
    This class processes audio frames to determine if they contain speech,
    helping to avoid processing silence and reduce unnecessary API calls.
    """
    def __init__(self, sample_rate=16000, frame_duration_ms=30, padding_duration_ms=300, aggressiveness=2):
        """Initialize the VAD with configurable parameters.
        
        Args:
            sample_rate: Audio sample rate (Hz)
            frame_duration_ms: Duration of each audio frame in milliseconds (10, 20, or 30)
            padding_duration_ms: Duration of padding to add around speech
            aggressiveness: VAD aggressiveness level (0-3, higher = more aggressive filtering)
        """
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.padding_duration_ms = padding_duration_ms
        self.vad = webrtcvad.Vad(aggressiveness)
        
        # Calculate frame size in samples
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        self.num_padding_frames = int(padding_duration_ms / frame_duration_ms)
        
        # Ring buffer for smoothing VAD decisions
        self.ring_buffer = collections.deque(maxlen=self.num_padding_frames)
        self.triggered = False
        
    def process_frame(self, frame):
        """Process a single audio frame and return speech detection result.
        
        Args:
            frame: Audio frame as bytes
            
        Returns:
            tuple: (is_speech, should_return_audio) where should_return_audio
                   indicates if accumulated audio should be returned
        """
        # Check if frame contains speech
        is_speech = self.vad.is_speech(frame, self.sample_rate)
        
        self.ring_buffer.append(1 if is_speech else 0)
        
        # We're in the triggered state when we've detected enough speech frames
        if not self.triggered:
            num_voiced = sum(self.ring_buffer)
            if num_voiced > 0.9 * self.ring_buffer.maxlen:
                self.triggered = True
                return is_speech, False
        else:
            # We're recording speech; check if it's ended
            num_unvoiced = len(self.ring_buffer) - sum(self.ring_buffer)
            if num_unvoiced > 0.9 * self.ring_buffer.maxlen:
                self.triggered = False
                return is_speech, True
                
        return is_speech, False
    
    def reset(self):
        """Reset the VAD state."""
        self.ring_buffer.clear()
        self.triggered = False

class SpeechRecognitionThread(QThread):
    """Enhanced QThread with VAD for continuous speech recognition."""
    
    recognized_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()
    audio_level_signal = pyqtSignal(float)  # For audio level visualization
    vad_status_signal = pyqtSignal(bool)    # For showing VAD status
    
    def __init__(self, recognizer, text_area_callback):
        super().__init__()
        self.recognizer = recognizer
        self.is_running = True
        self.text_area_callback = text_area_callback
        
        # Audio configuration
        self.sample_rate = 16000
        self.chunk_duration_ms = 30
        self.chunk_size = int(self.sample_rate * self.chunk_duration_ms / 1000)
        
        # Initialize VAD
        self.vad = VoiceActivityDetector(
            sample_rate=self.sample_rate,
            frame_duration_ms=self.chunk_duration_ms,
            padding_duration_ms=300,
            aggressiveness=2
        )
        
        # Initialize PyAudio for direct audio handling
        self.audio = pyaudio.PyAudio()
        
    def run(self):
        """Main execution with VAD-based speech detection."""
        stream = None
        try:
            # Open audio stream
            stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            self.text_area_callback("VAD initialized. Listening for speech...")
            
            # Buffer to accumulate audio frames
            audio_buffer = []
            
            while self.is_running:
                try:
                    # Read audio chunk
                    chunk = stream.read(self.chunk_size, exception_on_overflow=False)
                    
                    # Calculate and emit audio level for visualization
                    audio_data = np.frombuffer(chunk, dtype=np.int16)
                    audio_level = np.abs(audio_data).mean() / 32768.0
                    self.audio_level_signal.emit(audio_level)
                    
                    # Process frame with VAD
                    is_speech, should_process = self.vad.process_frame(chunk)
                    
                    # Emit VAD status for UI feedback
                    self.vad_status_signal.emit(self.vad.triggered)
                    
                    if self.vad.triggered:
                        audio_buffer.append(chunk)
                    
                    # When speech segment ends, process accumulated audio
                    if should_process and audio_buffer:
                        self.text_area_callback("Speech detected! Processing...")
                        
                        # Convert audio buffer to format expected by speech_recognition
                        audio_data = b''.join(audio_buffer)
                        
                        # Create AudioData object for speech recognition
                        audio = sr.AudioData(audio_data, self.sample_rate, 2)
                        
                        try:
                            # Recognize speech
                            recognized_text = self.recognizer.recognize_google(audio)
                            self.recognized_signal.emit(f"You said: {recognized_text}")
                        except sr.UnknownValueError:
                            self.recognized_signal.emit("Speech detected but could not understand")
                        except sr.RequestError as e:
                            self.recognized_signal.emit(f"API Error: {e}")
                            self.is_running = False
                        
                        # Clear buffer for next speech segment
                        audio_buffer = []
                        self.text_area_callback("Listening for speech...")
                        
                except Exception as e:
                    if self.is_running:  # Only report errors if we haven't stopped
                        self.recognized_signal.emit(f"Audio processing error: {e}")
                    
        except Exception as e:
            self.recognized_signal.emit(f"Failed to initialize audio: {e}")
        finally:
            if stream:
                stream.stop_stream()
                stream.close()
            self.finished_signal.emit()
            self.text_area_callback("Recording stopped.")
    
    def stop(self):
        """Stop the speech recognition loop."""
        self.is_running = False
        self.vad.reset()

class SpeechRecognitionApp(QMainWindow):
    """Enhanced main application window with VAD visualization."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Speech Recognition with Voice Activity Detection")
        self.setGeometry(100, 100, 700, 500)
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        self.layout = QVBoxLayout()
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Start Recording")
        self.start_button.clicked.connect(self.start_recording)
        button_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("Stop Recording")
        self.stop_button.clicked.connect(self.stop_recording)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)
        
        self.layout.addLayout(button_layout)
        
        # VAD status indicator
        status_layout = QHBoxLayout()
        status_layout.addWidget(QLabel("VAD Status:"))
        
        self.vad_status_label = QLabel("Inactive")
        self.vad_status_label.setStyleSheet("QLabel { background-color: gray; padding: 5px; border-radius: 3px; }")
        status_layout.addWidget(self.vad_status_label)
        
        status_layout.addStretch()
        self.layout.addLayout(status_layout)
        
        # Audio level indicator
        level_layout = QHBoxLayout()
        level_layout.addWidget(QLabel("Audio Level:"))
        
        self.audio_level_bar = QProgressBar()
        self.audio_level_bar.setMaximum(100)
        self.audio_level_bar.setTextVisible(False)
        self.audio_level_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid gray;
                border-radius: 3px;
                background-color: #f0f0f0;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 3px;
            }
        """)
        level_layout.addWidget(self.audio_level_bar)
        
        self.layout.addLayout(level_layout)
        
        # Text area for results
        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        self.layout.addWidget(self.text_area)
        
        self.central_widget.setLayout(self.layout)
        
        # Initialize speech recognizer
        self.recognizer = sr.Recognizer()
        self.recording_thread = None
        
        # Timer for updating audio level
        self.level_timer = QTimer()
        self.level_timer.timeout.connect(self.update_audio_level)
        self.current_audio_level = 0
        
        # ---------------- Virtual Camera Thread ---------------- #
        self.virtual_cam_thread = VirtualCameraThread()
        self.virtual_cam_thread.status_signal.connect(self.update_text_area)
        self.virtual_cam_thread.start()
        
    def start_recording(self):
        """Start the enhanced speech recognition process."""
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.text_area.append("Starting VAD-enhanced recording...\n")
        
        # Create and configure the speech recognition thread
        self.recording_thread = SpeechRecognitionThread(self.recognizer, self.update_text_area)
        self.recording_thread.recognized_signal.connect(self.display_recognized_text)
        self.recording_thread.finished_signal.connect(self.recording_finished)
        self.recording_thread.audio_level_signal.connect(self.set_audio_level)
        self.recording_thread.vad_status_signal.connect(self.update_vad_status)
        self.recording_thread.start()
        
        # Start audio level update timer
        self.level_timer.start(50)  # Update every 50ms
        
    def stop_recording(self):
        """Stop the speech recognition process."""
        if self.recording_thread and self.recording_thread.isRunning():
            self.recording_thread.stop()
        self.level_timer.stop()
        self.audio_level_bar.setValue(0)
        
    def recording_finished(self):
        """Clean up after recording finished."""
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.recording_thread = None
        self.vad_status_label.setText("Inactive")
        self.vad_status_label.setStyleSheet("QLabel { background-color: gray; padding: 5px; border-radius: 3px; }")
        
    def display_recognized_text(self, text):
        """Display recognized text."""
        self.update_text_area(f"{text}\n")
        
    def update_text_area(self, message):
        """Append message to text area."""
        self.text_area.append(message)
        
    def set_audio_level(self, level):
        """Update current audio level."""
        self.current_audio_level = int(level * 100)
        
    def update_audio_level(self):
        """Smoothly update audio level display."""
        current = self.audio_level_bar.value()
        target = self.current_audio_level
        new_value = current + (target - current) * 0.3
        self.audio_level_bar.setValue(int(new_value))
        
    def update_vad_status(self, is_active):
        """Update VAD status indicator."""
        if is_active:
            self.vad_status_label.setText("Speech Detected")
            self.vad_status_label.setStyleSheet("QLabel { background-color: #4CAF50; color: white; padding: 5px; border-radius: 3px; }")
        else:
            self.vad_status_label.setText("Listening...")
            self.vad_status_label.setStyleSheet("QLabel { background-color: #2196F3; color: white; padding: 5px; border-radius: 3px; }")

    def closeEvent(self, event):
        """Ensure all threads are properly stopped on window close."""
        try:
            if self.recording_thread and self.recording_thread.isRunning():
                self.recording_thread.stop()
                self.recording_thread.wait()
        except Exception:
            pass

        try:
            if self.virtual_cam_thread and self.virtual_cam_thread.isRunning():
                self.virtual_cam_thread.stop()
        except Exception:
            pass

        event.accept()

# ---------------------------------------------------------------------------
# Virtual camera thread (from virtual_camera_test.py)
# ---------------------------------------------------------------------------

class VirtualCameraThread(QThread):
    """QThread that captures real webcam frames, processes them with FrameHandler,
    and streams the result to a virtual camera using pyvirtualcam.
    """

    status_signal = pyqtSignal(str)  # Emit status text for UI

    def __init__(self, background_image: str = "likelion_hackathon.png", fps: int = 20):
        super().__init__()
        self.background_image = background_image
        self.fps = fps
        self._running = True

    def run(self):
        try:
            # Open physical webcam
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                self.status_signal.emit("Failed to open webcam")
                return

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            self.status_signal.emit(f"Webcam opened {width}x{height}")

            # Init frame handler and background
            handler = FrameHandler(width, height)
            try:
                handler.change_background(self.background_image)
            except Exception:
                # background may fail; ignore
                pass

            # Start virtual camera
            with pyvirtualcam.Camera(width=width, height=height, fps=self.fps) as cam:
                self.status_signal.emit(f"Virtual cam started → {cam.device}")

                while self._running:
                    ret, frame = cap.read()
                    if not ret:
                        self.status_signal.emit("Frame grab failed")
                        break

                    # Convert BGR to RGB for FrameHandler
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    processed = handler.handle_frame(frame_rgb)

                    # Send to virtual cam (expects RGB)
                    cam.send(processed)
                    cam.sleep_until_next_frame()

        except Exception as e:
            self.status_signal.emit(f"VirtualCam error: {e}")
        finally:
            try:
                cap.release()
            except Exception:
                pass

    def stop(self):
        self._running = False
        self.wait()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SpeechRecognitionApp()
    window.show()
    sys.exit(app.exec_())