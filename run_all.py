import subprocess
import sys
import os

base = os.path.dirname(os.path.abspath(__file__))

p1 = subprocess.Popen([sys.executable, os.path.join(base, "seedlink_stream.py")])
p2 = subprocess.Popen([sys.executable, os.path.join(base, "processing_service.py")])

p1.wait()
p2.wait()