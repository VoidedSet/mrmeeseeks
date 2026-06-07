import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtDBus import QDBusConnection, QDBusInterface
from PyQt6.QtCore import QObject, pyqtSlot, QTimer

class DBusReceiver(QObject):
    @pyqtSlot()
    def on_clicked(self):
        print("D-Bus Clicked signal received! ✓")
        QApplication.quit()

app = QApplication(sys.argv)
bus = QDBusConnection.sessionBus()
if not bus.isConnected():
    print("Cannot connect to session bus")
    sys.exit(1)

receiver = DBusReceiver()

# Connect to Clicked signal with service=None to receive from any sender
connected = bus.connect(
    None,                        # service (None means any sender)
    "/org/meeseeks/Pill",        # path
    "org.meeseeks.Pill",         # interface
    "Clicked",                   # name
    receiver.on_clicked          # decorated slot callable
)
print("Connected to Clicked signal:", connected)

# Call SetState to verify method calling
interface = QDBusInterface("org.meeseeks.Pill", "/org/meeseeks/Pill", "org.meeseeks.Pill", bus)
if interface.isValid():
    print("Interface is valid. Calling SetState to 'thinking'...")
    interface.call("SetState", "thinking")
else:
    print("Interface is invalid!")

# Add a timeout so it doesn't hang forever if not clicked
QTimer.singleShot(15000, lambda: (print("Timeout (15s) - Exiting."), QApplication.quit()))

print("Waiting for you to left-click the Meeseeks Pill in the GNOME status bar (or wait for 15s timeout)...")
app.exec()
