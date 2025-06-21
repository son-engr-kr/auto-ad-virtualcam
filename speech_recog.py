import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QTextEdit, QVBoxLayout, QWidget
from PyQt5.QtCore import QThread, pyqtSignal
import speech_recognition as sr

# Sets up Speech Recongition Thread to prevent GUI from freezing during speech recognition and inherits QThread to do so
class SpeechRecognitionThread(QThread):
    """A QThread that runs continuous speech recognition.

    This thread listens for audio from the microphone, processes it using
    the SpeechRecognition library, and emits signals with the recognized text
    or error messages. This prevents the main GUI from freezing during
    audio processing.

    Attributes:
        recognized_signal (pyqtSignal): Signal emitted with recognized text.
        finished_signal (pyqtSignal): Signal emitted when the thread finishes.
    """
    # Signal to send recognized text back to the main thread
    recognized_signal = pyqtSignal(str)
    # Signal to indicate the thread has finished its execution
    finished_signal = pyqtSignal()

    def __init__(self, recognizer, microphone, text_area_callback):
        """Initializes the SpeechRecognitionThread.

        Args:
            recognizer (sr.Recognizer): The speech recognition recognizer instance.
            microphone (sr.Microphone): The microphone instance for audio input.
            text_area_callback (callable): A function to call for updating the UI
                with status messages (e.g., "Listening...").
        """
        super().__init__()
        self.recognizer = recognizer
        self.microphone = microphone
        # Flag to control the running state of the recognition loop
        self.is_running = True
        # Callback function to update the GUI with status messages
        self.text_area_callback = text_area_callback

    def run(self):
        """The main execution method of the thread.

        Listens for audio in a loop, attempts to recognize speech, and emits
        the appropriate signals. The loop continues until `stop()` is called.
        """
        # Use the microphone as an audio source
        with self.microphone as source:
            # Calibrate the recognizer to the ambient noise level for better accuracy
            self.recognizer.adjust_for_ambient_noise(source)
            self.text_area_callback("Listening...")
            # Loop indefinitely until the stop() method is called
            while self.is_running:
                try:
                    # Listen for the next phrase from the microphone
                    audio = self.recognizer.listen(source, phrase_time_limit=10) # Listen for up to 10 seconds at a time
                    self.text_area_callback("Recognizing...")
                    # Use Google's speech recognition API to convert audio to text
                    recognized_text = self.recognizer.recognize_google(audio)
                    # Emit the recognized text to the main thread
                    self.recognized_signal.emit(f"You said: {recognized_text}")
                except sr.WaitTimeoutError:
                    # This error means no speech was detected within the timeout period; just continue listening
                    pass 
                except sr.UnknownValueError:
                    # This error means the speech was unintelligible
                    self.recognized_signal.emit("Could not understand audio")
                except sr.RequestError as e:
                    # This error is for network/API issues (e.g., no internet connection)
                    self.recognized_signal.emit(f"Could not request results; {e}")
                    self.is_running = False # Stop if there's a persistent error with the service
                if not self.is_running:
                    break
        # Emit a signal to indicate that the thread has finished
        self.finished_signal.emit()
        self.text_area_callback("Recording stopped.")

    def stop(self):
        """Stops the speech recognition loop."""
        # Set the flag to False to break the while loop in the run() method
        self.is_running = False

class SpeechRecognitionApp(QMainWindow):
    """Main application window for continuous speech recognition.

    This class sets up the user interface with start/stop buttons and a
    text area to display results. It manages the `SpeechRecognitionThread`
    for handling the audio processing.
    """
    def __init__(self):
        """Initializes the application's UI and state."""
        super().__init__()
        self.setWindowTitle("Continuous Speech Recognition")
        self.setGeometry(100, 100, 600, 400)

        # Main widget to hold the layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Vertical layout for arranging widgets
        self.layout = QVBoxLayout()

        self.start_button = QPushButton("Start Recording")
        # Connect the button's clicked signal to the start_recording method
        self.start_button.clicked.connect(self.start_recording)
        self.layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop Recording")
        self.stop_button.clicked.connect(self.stop_recording)
        # The stop button is disabled until recording starts
        self.stop_button.setEnabled(False) 
        self.layout.addWidget(self.stop_button)

        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True) # Make the text area non-editable by the user
        self.layout.addWidget(self.text_area)

        self.central_widget.setLayout(self.layout)

        # Initialize the speech recognizer and microphone
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.recording_thread = None

    def start_recording(self):
        """Starts the speech recognition process.

        Disables the start button, enables the stop button, and starts the
        `SpeechRecognitionThread`.
        """
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.text_area.append("Starting recording...\n")
        # Create and configure the speech recognition thread
        self.recording_thread = SpeechRecognitionThread(self.recognizer, self.microphone, self.update_text_area)
        # Connect the thread's signals to the main window's methods (slots)
        self.recording_thread.recognized_signal.connect(self.display_recognized_text)
        self.recording_thread.finished_signal.connect(self.recording_finished)
        # Start the thread's execution by calling its run() method
        self.recording_thread.start()

    def stop_recording(self):
        """Stops the speech recognition process.

        Signals the `SpeechRecognitionThread` to stop its execution loop.
        """
        if self.recording_thread and self.recording_thread.isRunning():
            # Call the thread's stop method to safely terminate the loop
            self.recording_thread.stop()

    def recording_finished(self):
        """Cleans up after the recording thread has finished.

        Resets the button states and clears the thread reference.
        """
        # Reset the UI buttons to their initial state
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        # Clear the reference to the finished thread
        self.recording_thread = None

    def display_recognized_text(self, text):
        """Displays the recognized text in the text area.

        Args:
            text (str): The text that was recognized from the audio.
        """
        self.update_text_area(f"{text}\n")

    def update_text_area(self, message):
        """Appends a message to the text area.

        Args:
            message (str): The message to append to the text area.
        """
        self.text_area.append(message)

# This block ensures the code runs only when the script is executed directly
if __name__ == '__main__':
    # Create the PyQt application instance
    app = QApplication(sys.argv)
    # Create an instance of our main application window
    window = SpeechRecognitionApp()
    # Show the window on the screen
    window.show()
    # Start the application's event loop and exit when it's done
    sys.exit(app.exec_())