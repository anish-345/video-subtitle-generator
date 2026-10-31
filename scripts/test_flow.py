import urllib.request
import urllib.parse
import json
import time
import os
import sys
import subprocess
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import FFMPEG_PATH

# 1. Create a 5s test file named 'my_speech_demo.mp4'
test_file = Path('data') / 'temp' / 'my_speech_demo.mp4'
test_file.parent.mkdir(parents=True, exist_ok=True)
cmd = [
    FFMPEG_PATH, '-y', '-f', 'lavfi', '-i', 'sine=frequency=1000:duration=5',
    '-f', 'lavfi', '-i', 'color=c=black:s=320x240:d=5',
    '-c:v', 'libx264', '-c:a', 'aac', str(test_file)
]
subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print('Created test file:', test_file.name, 'Exists:', test_file.exists())

# 2. Upload via Rust Gateway multipart
boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
body = bytearray()
body.extend(f'--{boundary}\r\n'.encode())
body.extend(f'Content-Disposition: form-data; name="file"; filename="my_speech_demo.mp4"\r\n'.encode())
body.extend(b'Content-Type: video/mp4\r\n\r\n')
body.extend(test_file.read_bytes())
body.extend(b'\r\n')

for field, val in [('model_size', 'base'), ('language', 'en'), ('task', 'transcribe'), ('turbo_mode', 'true'), ('vad_filter', 'false')]:
    body.extend(f'--{boundary}\r\n'.encode())
    body.extend(f'Content-Disposition: form-data; name="{field}"\r\n\r\n'.encode())
    body.extend(f'{val}\r\n'.encode())
body.extend(f'--{boundary}--\r\n'.encode())

req = urllib.request.Request(
    'http://127.0.0.1:8000/api/transcribe',
    data=bytes(body),
    headers={'Content-Type': f'multipart/form-data; boundary={boundary}'}
)

with urllib.request.urlopen(req) as resp:
    init_res = json.loads(resp.read())
print('Upload response:', init_res)
job_id = init_res['job_id']

# 3. Poll status until complete
for _ in range(60):
    time.sleep(1)
    with urllib.request.urlopen(f'http://127.0.0.1:8000/api/jobs/{job_id}/status') as s_resp:
        status_data = json.loads(s_resp.read())
        print(f'Job {job_id}: {status_data["status"]} - {status_data.get("progress", 0)}% - {status_data.get("message", "")[:60]}')
        if status_data['status'] in ['completed', 'failed']:
            break

# 4. Verify video auto-deletion
uploaded_videos = list(Path('data/uploads').glob(f'{job_id}_*'))
print(f'Uploaded video in data/uploads: {uploaded_videos} (Should be EMPTY / deleted!)')

# 5. Verify download headers and filename
download_req = urllib.request.Request(f'http://127.0.0.1:8000/api/jobs/{job_id}/download/srt')
with urllib.request.urlopen(download_req) as dl_resp:
    print('Download Status:', dl_resp.status)
    print('Content-Disposition:', dl_resp.headers.get('Content-Disposition'))
