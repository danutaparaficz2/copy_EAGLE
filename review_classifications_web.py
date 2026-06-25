"""
Web-based GUI tool to review image classifications.
Access at: http://localhost:8001
"""
import os
import csv
import json
import base64
from pathlib import Path
from flask import Flask, jsonify, request
from PIL import Image
import io

app = Flask(__name__)

# Configuration
PANEL = "23-P09-C"
MODE = "EL"
IMAGE_FOLDER = Path(f"normalized_images/{PANEL}/{MODE}")
LABELS_CSV = Path(f"OPENAI/{PANEL}/classification_results_{MODE}_{PANEL}.csv")
APPROVED_CSV = Path(f"OPENAI/{PANEL}/approved_classifications_{MODE}_{PANEL}.csv")
WRONG_CSV = Path(f"OPENAI/{PANEL}/wrong_classifications_{MODE}_{PANEL}.csv")
NOT_SURE_CSV = Path(f"OPENAI/{PANEL}/not_sure_classifications_{MODE}_{PANEL}.csv")

# Max image display size
MAX_IMG_SIZE = (640, 640)

# Initialize CSV files with headers
for csv_path in [APPROVED_CSV, WRONG_CSV, NOT_SURE_CSV]:
    if not csv_path.exists():
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["filename", "labels"])


def load_csv(csv_path):
    """Load (filename, labels) pairs from CSV."""
    rows = []
    if not csv_path.exists():
        return rows
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # Skip header
        for row in reader:
            if len(row) >= 2:
                rows.append((row[0].strip(), row[1].strip()))
            elif len(row) == 1:
                rows.append((row[0].strip(), ""))
    return rows


def load_reviewed_set(csv_path):
    """Return set of filenames already reviewed."""
    reviewed = set()
    if not csv_path.exists():
        return reviewed
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # Skip header
        for row in reader:
            if row:
                reviewed.add(row[0].strip())
    return reviewed


def get_pending_images():
    """Get list of images that haven't been reviewed yet."""
    all_entries = load_csv(LABELS_CSV)
    approved = load_reviewed_set(APPROVED_CSV)
    wrong = load_reviewed_set(WRONG_CSV)
    not_sure = load_reviewed_set(NOT_SURE_CSV)
    done = approved | wrong | not_sure
    
    pending = [(f, l) for f, l in all_entries if f not in done]
    pending.sort(key=lambda x: (x[1].strip().lower() == 'good', x[0]))
    return pending


def image_to_base64(img_path):
    """Convert image to base64 for embedding in HTML."""
    if not img_path.exists():
        return None
    try:
        img = Image.open(img_path).convert("RGB")
        img.thumbnail(MAX_IMG_SIZE, Image.LANCZOS)
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/jpeg;base64,{img_str}"
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        return None


# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Classification Review</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #212121;
            color: #ffffff;
            padding: 20px;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .progress {
            font-size: 14px;
            color: #aaaaaa;
            margin-bottom: 10px;
        }
        .title {
            font-size: 28px;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .info {
            background: #1a1a1a;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            text-align: center;
        }
        .filename {
            font-size: 16px;
            font-weight: bold;
            word-break: break-all;
            margin-bottom: 8px;
        }
        .label {
            font-size: 14px;
            color: #aaaaaa;
        }
        .image-container {
            background: #1a1a1a;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            display: flex;
            justify-content: center;
            min-height: 400px;
            align-items: center;
        }
        .image-container img {
            max-width: 100%;
            max-height: 600px;
            border-radius: 4px;
        }
        .image-placeholder {
            color: #ff5555;
            font-size: 18px;
            text-align: center;
        }
        .button-group {
            display: flex;
            gap: 15px;
            margin-bottom: 20px;
            flex-wrap: wrap;
            justify-content: center;
        }
        button {
            padding: 12px 24px;
            font-size: 16px;
            font-weight: bold;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            transition: background-color 0.2s;
            min-width: 140px;
        }
        .btn-correct {
            background: #2e7d32;
            color: white;
        }
        .btn-correct:hover { background: #1b5e20; }
        .btn-correct:disabled { background: #555555; cursor: not-allowed; }
        
        .btn-wrong {
            background: #c62828;
            color: white;
        }
        .btn-wrong:hover { background: #7f0000; }
        .btn-wrong:disabled { background: #555555; cursor: not-allowed; }
        
        .btn-not-sure {
            background: #e65100;
            color: white;
        }
        .btn-not-sure:hover { background: #bf360c; }
        .btn-not-sure:disabled { background: #555555; cursor: not-allowed; }
        
        .nav-group {
            display: flex;
            gap: 15px;
            justify-content: center;
            margin-bottom: 20px;
        }
        .btn-nav {
            background: #37474f;
            color: white;
            min-width: 120px;
        }
        .btn-nav:hover { background: #263238; }
        .btn-nav:disabled { background: #555555; cursor: not-allowed; }
        
        .btn-reset {
            background: #5d4037;
            color: white;
            min-width: 100px;
        }
        .btn-reset:hover { background: #3e2723; }
        
        .hint {
            text-align: center;
            color: #666666;
            font-size: 12px;
            margin-top: 15px;
        }
        .status {
            text-align: center;
            padding: 10px;
            border-radius: 4px;
            margin-bottom: 20px;
            font-size: 14px;
        }
        .status.done {
            background: #1b5e20;
            color: #4caf50;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 10px;
            margin-top: 20px;
        }
        .stat-box {
            background: #1a1a1a;
            padding: 15px;
            border-radius: 4px;
            text-align: center;
        }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #4caf50;
        }
        .stat-label {
            font-size: 12px;
            color: #aaaaaa;
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="progress"><span id="progress">Loading...</span></div>
            <div class="title">Classification Review</div>
            <div style="font-size: 12px; color: #666666;">Panel: {{ panel }} | Mode: {{ mode }}</div>
        </div>

        <div id="status-message" class="status" style="display:none;"></div>

        <div class="info">
            <div class="filename" id="filename">-</div>
            <div class="label" id="label-text">-</div>
        </div>

        <div class="image-container">
            <img id="image" style="display:none;" />
            <div id="image-placeholder" class="image-placeholder">Loading image...</div>
        </div>

        <div class="button-group">
            <button class="btn-correct" id="btn-correct" onclick="classifyImage('correct')">✓ Correct</button>
            <button class="btn-wrong" id="btn-wrong" onclick="classifyImage('wrong')">✗ Wrong</button>
            <button class="btn-not-sure" id="btn-not-sure" onclick="classifyImage('not_sure')">? Not Sure</button>
        </div>

        <div class="nav-group">
            <button class="btn-nav" id="btn-prev" onclick="navigate('prev')">◀ Previous</button>
            <button class="btn-nav" id="btn-next" onclick="navigate('next')">Next ▶</button>
        </div>

        <div style="text-align: center;">
            <button class="btn-reset" onclick="resetReview()">↺ Reset All</button>
        </div>

        <div class="stats">
            <div class="stat-box">
                <div class="stat-value" id="stat-pending">-</div>
                <div class="stat-label">Pending</div>
            </div>
            <div class="stat-box">
                <div class="stat-value" id="stat-approved">-</div>
                <div class="stat-label">Approved</div>
            </div>
            <div class="stat-box">
                <div class="stat-value" id="stat-wrong">-</div>
                <div class="stat-label">Wrong</div>
            </div>
            <div class="stat-box">
                <div class="stat-value" id="stat-not-sure">-</div>
                <div class="stat-label">Not Sure</div>
            </div>
        </div>

        <div class="hint" style="margin-top: 30px;">
            Keyboard: Enter=Correct | Backspace=Wrong | N=Not Sure | ←/→=Navigate | R=Reset
        </div>
    </div>

    <script>
        let currentIndex = 0;
        let pendingImages = [];

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') classifyImage('correct');
            else if (e.key === 'Backspace') classifyImage('wrong');
            else if (e.key === 'n' || e.key === 'N') classifyImage('not_sure');
            else if (e.key === 'ArrowLeft') navigate('prev');
            else if (e.key === 'ArrowRight') navigate('next');
            else if (e.key === 'r' || e.key === 'R') resetReview();
        });

        async function loadPendingImages() {
            try {
                const response = await fetch('/api/pending');
                pendingImages = await response.json();
                currentIndex = 0;
                displayImage();
                updateStats();
            } catch (error) {
                console.error('Error loading images:', error);
            }
        }

        async function displayImage() {
            if (pendingImages.length === 0) {
                showDoneMessage();
                disableButtons();
                return;
            }

            if (currentIndex >= pendingImages.length) {
                showDoneMessage();
                disableButtons();
                return;
            }

            enableButtons();
            const [filename, label] = pendingImages[currentIndex];
            
            document.getElementById('filename').textContent = filename;
            document.getElementById('label-text').textContent = label || '(no label)';
            document.getElementById('progress').textContent = `${currentIndex + 1} / ${pendingImages.length}`;

            try {
                const response = await fetch(`/api/image?filename=${encodeURIComponent(filename)}`);
                const data = await response.json();
                
                if (data.image) {
                    const img = document.getElementById('image');
                    img.src = data.image;
                    img.style.display = 'block';
                    document.getElementById('image-placeholder').style.display = 'none';
                } else {
                    document.getElementById('image-placeholder').textContent = '[Image not found]';
                    document.getElementById('image').style.display = 'none';
                    document.getElementById('image-placeholder').style.display = 'block';
                }
            } catch (error) {
                console.error('Error loading image:', error);
                document.getElementById('image-placeholder').textContent = '[Error loading image]';
                document.getElementById('image').style.display = 'none';
                document.getElementById('image-placeholder').style.display = 'block';
            }

            updateNavButtons();
        }

        function updateNavButtons() {
            document.getElementById('btn-prev').disabled = currentIndex <= 0;
            document.getElementById('btn-next').disabled = currentIndex >= pendingImages.length - 1;
        }

        function enableButtons() {
            document.getElementById('btn-correct').disabled = false;
            document.getElementById('btn-wrong').disabled = false;
            document.getElementById('btn-not-sure').disabled = false;
        }

        function disableButtons() {
            document.getElementById('btn-correct').disabled = true;
            document.getElementById('btn-wrong').disabled = true;
            document.getElementById('btn-not-sure').disabled = true;
        }

        async function classifyImage(classification) {
            if (currentIndex >= pendingImages.length) return;

            const [filename, label] = pendingImages[currentIndex];

            try {
                const response = await fetch('/api/classify', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ filename, label, classification })
                });

                if (response.ok) {
                    pendingImages.splice(currentIndex, 1);
                    if (currentIndex >= pendingImages.length && pendingImages.length > 0) {
                        currentIndex = pendingImages.length - 1;
                    }
                    updateStats();
                    displayImage();
                }
            } catch (error) {
                console.error('Error classifying image:', error);
            }
        }

        function navigate(direction) {
            if (direction === 'prev' && currentIndex > 0) {
                currentIndex--;
            } else if (direction === 'next' && currentIndex < pendingImages.length - 1) {
                currentIndex++;
            }
            displayImage();
        }

        function showDoneMessage() {
            const msg = document.getElementById('status-message');
            msg.textContent = '✓ All pending images reviewed!';
            msg.classList.add('done');
            msg.style.display = 'block';
            document.getElementById('filename').textContent = 'All Done!';
            document.getElementById('label-text').textContent = '';
            document.getElementById('image').style.display = 'none';
            document.getElementById('image-placeholder').textContent = '✓';
            document.getElementById('image-placeholder').style.display = 'block';
            document.getElementById('image-placeholder').style.fontSize = '60px';
        }

        async function updateStats() {
            try {
                const response = await fetch('/api/stats');
                const stats = await response.json();
                document.getElementById('stat-pending').textContent = pendingImages.length;
                document.getElementById('stat-approved').textContent = stats.approved;
                document.getElementById('stat-wrong').textContent = stats.wrong;
                document.getElementById('stat-not-sure').textContent = stats.not_sure;
            } catch (error) {
                console.error('Error updating stats:', error);
            }
        }

        async function resetReview() {
            if (!confirm('Delete all review records and start over?')) return;

            try {
                const response = await fetch('/api/reset', { method: 'POST' });
                if (response.ok) {
                    loadPendingImages();
                    document.getElementById('status-message').style.display = 'none';
                }
            } catch (error) {
                console.error('Error resetting:', error);
            }
        }

        // Load images on page load
        loadPendingImages();
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    html = HTML_TEMPLATE.replace('{{ panel }}', PANEL).replace('{{ mode }}', MODE)
    return html


@app.route('/api/pending')
def api_pending():
    """Get list of pending images."""
    pending = get_pending_images()
    return jsonify(pending)


@app.route('/api/image')
def api_image():
    """Get image as base64."""
    filename = request.args.get('filename', '')
    img_path = IMAGE_FOLDER / filename
    img_b64 = image_to_base64(img_path)
    return jsonify({'image': img_b64})


@app.route('/api/classify', methods=['POST'])
def api_classify():
    """Classify an image."""
    data = request.get_json()
    filename = data.get('filename', '')
    label = data.get('label', '')
    classification = data.get('classification', '')

    csv_path = {
        'correct': APPROVED_CSV,
        'wrong': WRONG_CSV,
        'not_sure': NOT_SURE_CSV,
    }.get(classification)

    if csv_path:
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([filename, label])
        print(f"[{classification.upper()}] {filename:<50} {label}")
        return jsonify({'status': 'ok'})
    
    return jsonify({'error': 'Invalid classification'}), 400


@app.route('/api/stats')
def api_stats():
    """Get review statistics."""
    all_count = len(load_csv(LABELS_CSV))
    approved = len(load_reviewed_set(APPROVED_CSV))
    wrong = len(load_reviewed_set(WRONG_CSV))
    not_sure = len(load_reviewed_set(NOT_SURE_CSV))
    
    return jsonify({
        'total': all_count,
        'approved': approved,
        'wrong': wrong,
        'not_sure': not_sure,
        'pending': all_count - (approved + wrong + not_sure)
    })


@app.route('/api/reset', methods=['POST'])
def api_reset():
    """Reset all classifications."""
    for csv_path in [APPROVED_CSV, WRONG_CSV, NOT_SURE_CSV]:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(['filename', 'labels'])
    print("[RESET] Cleared all review records")
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    print(f"\n{'='*60}")
    print(f"Classification Review Web Interface")
    print(f"{'='*60}")
    print(f"\nPanel: {PANEL}")
    print(f"Mode: {MODE}")
    print(f"Images: {IMAGE_FOLDER}")
    print(f"\nAccess at: http://localhost:8001")
    print(f"\nPress Ctrl+C to stop\n")
    
    app.run(host='0.0.0.0', port=8001, debug=False)
