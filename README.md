<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>フォーム比較＆骨格解析アプリ</title>
    
    <!-- 骨格検出用のAIライブラリ (TensorFlow.js & Pose Detection) -->
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs"></script>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/pose-detection"></script>

    <!-- スタイルシート (旧 style.css) -->
    <style>
        * {
            box-sizing: border-box;
        }
        body {
            margin: 0;
            padding: 0;
            background-color: #1a1a1a;
            color: #fff;
            font-family: sans-serif;
            display: flex;
            flex-direction: column;
            height: 100vh;
            overflow: hidden;
        }
        .controls {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 20px;
            background-color: #333;
            font-size: 14px;
        }
        .control-group {
            flex: 1;
        }
        .control-group.center {
            text-align: center;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 8px;
        }
        input[type="number"] {
            width: 60px;
        }
        button {
            padding: 10px 20px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
        }
        .video-area {
            display: flex;
            flex: 1;
            width: 100%;
        }
        .video-wrapper {
            position: relative;
            flex: 1;
            border: 1px solid #444;
            background: #000;
        }
        video {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }
        canvas {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: contain;
            pointer-events: none;
        }
    </style>
</head>
<body>
    <!-- 上部の操作パネル -->
    <div class="controls">
        <div class="control-group">
            <label>動画1: <input type="file" id="upload1" accept="video/*"></label><br>
            <label>開始: <input type="number" id="start1" value="0" step="0.1" min="0">秒</label>
        </div>
        <div class="control-group center">
            <button id="playBtn">同時再生 / リセット</button>
            <label>速度: <input type="range" id="speed" min="0.1" max="1" step="0.1" value="1"> <span id="speedVal">1.0</span>x</label>
        </div>
        <div class="control-group">
            <label>動画2: <input type="file" id="upload2" accept="video/*"></label><br>
            <label>開始: <input type="number" id="start2" value="0" step="0.1" min="0">秒</label>
        </div>
    </div>

    <!-- 左右の動画表示エリア -->
    <div class="video-area">
        <div class="video-wrapper">
            <video id="video1" playsinline muted></video>
            <canvas id="canvas1"></canvas>
        </div>
        <div class="video-wrapper">
            <video id="video2" playsinline muted></video>
            <canvas id="canvas2"></canvas>
        </div>
    </div>

    <!-- 動作スクリプト (旧 script.js) -->
    <script>
        const v1 = document.getElementById('video1');
        const v2 = document.getElementById('video2');
        const c1 = document.getElementById('canvas1');
        const c2 = document.getElementById('canvas2');
        const ctx1 = c1.getContext('2d');
        const ctx2 = c2.getContext('2d');

        let detector;
        let isPlaying = false;

        async function initAI() {
            const detectorConfig = { modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING };
            detector = await poseDetection.createDetector(poseDetection.SupportedModels.MoveNet, detectorConfig);
            console.log("AIモデルの読み込み完了");
        }
        initAI();

        document.getElementById('upload1').addEventListener('change', e => loadVideo(e, v1));
        document.getElementById('upload2').addEventListener('change', e => loadVideo(e, v2));

        function loadVideo(event, videoEl) {
            const file = event.target.files[0];
            if (file) {
                videoEl.src = URL.createObjectURL(file);
                videoEl.load();
            }
        }

        document.getElementById('speed').addEventListener('input', e => {
            const speed = parseFloat(e.target.value);
            v1.playbackRate = speed;
            v2.playbackRate = speed;
            document.getElementById('speedVal').textContent = speed.toFixed(1);
        });

        document.getElementById('playBtn').addEventListener('click', () => {
            if (isPlaying) {
                v1.pause();
                v2.pause();
            } else {
                v1.currentTime = parseFloat(document.getElementById('start1').value);
                v2.currentTime = parseFloat(document.getElementById('start2').value);
                
                v1.play();
                v2.play();
                detectAndDraw();
            }
            isPlaying = !isPlaying;
        });

        async function detectAndDraw() {
            if (!isPlaying) return;
            await processFrame(v1, c1, ctx1);
            await processFrame(v2, c2, ctx2);
            requestAnimationFrame(detectAndDraw);
        }

        async function processFrame(video, canvas, ctx) {
            if (video.readyState < 2 || !detector) return;
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            const poses = await detector.estimatePoses(video);
            if (poses.length > 0) drawSkeleton(poses[0].keypoints, ctx);
        }

        function drawSkeleton(keypoints, ctx) {
            const threshold = 0.3;
            const keypointMap = {};

            keypoints.forEach(p => {
                keypointMap[p.name] = p;
                if (p.score > threshold) {
                    ctx.beginPath();
                    ctx.arc(p.x, p.y, 6, 0, 2 * Math.PI);
                    ctx.fillStyle = '#00FF00';
                    ctx.fill();
                }
            });

            const edges = [
                ['left_shoulder', 'right_shoulder'],
                ['left_shoulder', 'left_elbow'], ['left_elbow', 'left_wrist'],
                ['right_shoulder', 'right_elbow'], ['right_elbow', 'right_wrist'],
                ['left_shoulder', 'left_hip'], ['right_shoulder', 'right_hip'],
                ['left_hip', 'right_hip'],
                ['left_hip', 'left_knee'], ['left_knee', 'left_ankle'],
                ['right_hip', 'right_knee'], ['right_knee', 'right_ankle']
            ];

            ctx.lineWidth = 4;
            ctx.strokeStyle = '#FF0000';
            edges.forEach(([p1, p2]) => {
                const kp1 = keypointMap[p1];
                const kp2 = keypointMap[p2];
                if (kp1 && kp2 && kp1.score > threshold && kp2.score > threshold) {
                    ctx.beginPath();
                    ctx.moveTo(kp1.x, kp1.y);
                    ctx.lineTo(kp2.x, kp2.y);
                    ctx.stroke();
                }
            });
        }
    </script>
</body>
</html>
