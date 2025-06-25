const express = require('express');
const ffmpeg = require('fluent-ffmpeg');
const path = require('path');
const fs = require('fs');
const cors = require('cors');

const app = express();
const PORT = 5001;

app.use(cors());
app.use('/hls', express.static(path.join(__dirname, 'hls')));

const hlsDir = path.join(__dirname, 'hls');
if (!fs.existsSync(hlsDir)) {
    fs.mkdirSync(hlsDir, { recursive: true });
}

function startHLSStream() {
    const outputPath = path.join(hlsDir, 'stream.m3u8');
    
    ffmpeg()
        .input('video="Integrated Camera"')
        .inputFormat('dshow')
        .videoCodec('libx264')
        .addOption('-preset', 'fast')
        .addOption('-tune', 'zerolatency')
        .addOption('-f', 'hls')
        .addOption('-hls_time', '2')
        .addOption('-hls_list_size', '3')
        .addOption('-hls_flags', 'delete_segments')
        .output(outputPath)
        .on('start', (cmd) => console.log('FFmpeg started:', cmd))
        .on('error', (err) => {
            console.error('FFmpeg error:', err.message);
            setTimeout(startHLSStream, 5000);
        })
        .run();
}

app.get('/api/stream/status', (req, res) => {
    const streamFile = path.join(hlsDir, 'stream.m3u8');
    res.json({
        streaming: fs.existsSync(streamFile),
        streamUrl: `http://localhost:${PORT}/hls/stream.m3u8`
    });
});

app.listen(PORT, () => {
    console.log(`🎥 Stream server: http://localhost:${PORT}`);
    setTimeout(startHLSStream, 2000);
});