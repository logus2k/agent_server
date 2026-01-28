# Voice Services Integration Guide

## Overview

This guide documents how to integrate Speech-to-Text (STT), Language Model (LLM), and Text-to-Speech (TTS) services into web applications. The architecture uses Socket.IO for real-time communication between browser clients and backend services.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              BROWSER                                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                  │
│  │  Microphone │    │  AgentClient│    │  TTS Audio  │                  │
│  │  (AudioCtx) │    │  (Socket.IO)│    │  Playback   │                  │
│  └──────┬──────┘    └──────┬──────┘    └──────▲──────┘                  │
│         │                  │                  │                          │
└─────────┼──────────────────┼──────────────────┼──────────────────────────┘
          │                  │                  │
          │ audio_data       │ JoinSTT/Chat     │ tts_audio_chunk
          │                  │ UserTranscript   │
          ▼                  ▼ ChatChunk        │
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   STT Service   │  │  Agent Server   │  │   TTS Service   │
│   (port 2700)   │◄─┤   (LLM Host)    ├─►│   (port 7766)   │
│                 │  │                 │  │                 │
│ - Whisper model │  │ - llama.cpp     │  │ - Kokoro/other  │
│ - Transcription │  │ - Agent presets │  │ - Audio synth   │
└─────────────────┘  │ - Memory        │  └─────────────────┘
                     │ - TTS routing   │
                     └─────────────────┘
```

## Service Endpoints

| Service | URL | Socket.IO Path |
|---------|-----|----------------|
| STT | `https://your-domain.com/stt` | `/stt/socket.io` |
| Agent Server (LLM) | `https://your-domain.com/llm` | `/llm/socket.io` |
| TTS | `https://your-domain.com/tts` | `/tts/socket.io` |

---

## Part 1: Agent Client Setup

### 1.1 AgentClient Class

The `AgentClient` class manages the connection to the agent server for LLM interactions.

```javascript
// agentClient.js
export class AgentClient {
    constructor(options = {}) {
        this.url = options.url || 'https://your-domain.com/llm';
        this.path = options.path || '/llm/socket.io';
        this.socket = null;
        this._streamHandlers = {};
        this._transcriptHandlers = {};
    }

    async connect(options = {}) {
        return new Promise((resolve, reject) => {
            this.socket = io(this.url, {
                path: this.path,
                transports: ['websocket', 'polling'],
                reconnection: true,
                reconnectionAttempts: 5,
                reconnectionDelay: 1000
            });

            this.socket.once('connect', () => {
                this._setupEventHandlers();
                resolve();
            });

            this.socket.once('connect_error', reject);
        });
    }

    _setupEventHandlers() {
        // LLM streaming events
        this.socket.on('RunStarted', (data) => {
            this._streamHandlers.onStarted?.(data.runId);
        });

        this.socket.on('ChatChunk', (data) => {
            // Accumulate chunks for full response
            this._currentResponse = (this._currentResponse || '') + data.chunk;
            this._streamHandlers.onText?.(this._currentResponse);
        });

        this.socket.on('ChatDone', (data) => {
            this._streamHandlers.onDone?.(data.runId);
            this._currentResponse = '';
        });

        this.socket.on('Error', (data) => {
            this._streamHandlers.onError?.(data);
        });

        // STT transcript events (from agent server)
        this.socket.on('UserTranscript', (data) => {
            if (data.final) {
                this._transcriptHandlers.onFinal?.(data);
            } else {
                this._transcriptHandlers.onInterim?.(data);
            }
        });
    }

    // Register stream handlers
    onStream(handlers) {
        this._streamHandlers = handlers;
    }

    // Register transcript handlers
    onTranscripts(handlers) {
        this._transcriptHandlers = handlers;
    }

    // Send text to LLM
    async chat(text, options = {}) {
        return new Promise((resolve, reject) => {
            this.socket.emit('Chat', {
                text,
                agent: options.agent || 'default',
                thread_id: options.threadId,
                memory: options.memory || 'none'
            }, (ack) => {
                if (ack?.error) reject(new Error(ack.error));
                else resolve(ack);
            });
        });
    }

    // Subscribe agent server to STT transcripts
    async sttSubscribe({ sttUrl, clientId, agent, threadId }) {
        return new Promise((resolve, reject) => {
            this.socket.emit('JoinSTT', {
                sttUrl,
                clientId,
                agent,
                threadId
            }, (ack) => {
                if (ack?.error) reject(new Error(ack.error));
                else resolve(ack);
            });
        });
    }

    // Unsubscribe from STT
    async sttUnsubscribe({ sttUrl, clientId }) {
        return new Promise((resolve) => {
            this.socket.emit('LeaveSTT', { sttUrl, clientId }, resolve);
        });
    }

    // Subscribe to TTS
    async ttsSubscribe({ clientId, voice, speed }) {
        return new Promise((resolve, reject) => {
            this.socket.emit('JoinTTS', {
                clientId,
                voice,
                speed
            }, (ack) => {
                if (ack?.error) reject(new Error(ack.error));
                else resolve(ack);
            });
        });
    }

    // Unsubscribe from TTS
    async ttsUnsubscribe({ clientId }) {
        return new Promise((resolve) => {
            this.socket.emit('LeaveTTS', { clientId }, resolve);
        });
    }

    // Cancel current LLM generation
    cancel() {
        this.socket?.emit('Interrupt');
    }

    disconnect() {
        this.socket?.disconnect();
        this.socket = null;
    }
}
```

---

## Part 2: Speech-to-Text (STT) Integration

### 2.1 Audio Capture Setup

```javascript
// Audio configuration
const SAMPLE_RATE = 48000;      // Browser AudioContext sample rate
const TARGET_RATE = 16000;      // STT expected sample rate
const PACKET_MS = 100;          // Send audio every 100ms

// State variables
let audioContext = null;
let mediaStream = null;
let workletNode = null;
let sttSocket = null;
let resampler = null;
```

### 2.2 Audio Worklet (recorder_worklet.js)

```javascript
// recorder_worklet.js
class RecorderWorklet extends AudioWorkletProcessor {
    process(inputs) {
        const input = inputs[0];
        if (input && input[0]) {
            // Send float32 audio data to main thread
            this.port.postMessage(new Float32Array(input[0]));
        }
        return true;
    }
}

registerProcessor('recorder-worklet', RecorderWorklet);
```

### 2.3 Audio Resampler

```javascript
// audioResampler.js
export class AudioResampler {
    constructor(inputRate, outputRate) {
        this.inputRate = inputRate;
        this.outputRate = outputRate;
        this.ratio = outputRate / inputRate;
    }

    pushFloat32(float32Array) {
        // Simple linear interpolation resampling
        const outputLength = Math.floor(float32Array.length * this.ratio);
        const output = new Int16Array(outputLength);
        
        for (let i = 0; i < outputLength; i++) {
            const srcIndex = i / this.ratio;
            const srcIndexFloor = Math.floor(srcIndex);
            const srcIndexCeil = Math.min(srcIndexFloor + 1, float32Array.length - 1);
            const t = srcIndex - srcIndexFloor;
            
            // Interpolate
            const sample = float32Array[srcIndexFloor] * (1 - t) + 
                          float32Array[srcIndexCeil] * t;
            
            // Convert to int16
            output[i] = Math.max(-32768, Math.min(32767, Math.floor(sample * 32767)));
        }
        
        return output;
    }

    reset() {
        // Reset any internal state if needed
    }
}
```

### 2.4 Starting STT

```javascript
async function startSTT(clientId, agentClient, agentName, threadId) {
    const STT_URL = 'https://your-domain.com/stt';
    const STT_PATH = '/stt/socket.io';

    // 1. Request microphone access
    mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
            channelCount: 1,
            sampleRate: SAMPLE_RATE,
            echoCancellation: true,
            noiseSuppression: true
        }
    });

    // 2. Set up AudioContext and Worklet
    audioContext = new AudioContext({ sampleRate: SAMPLE_RATE });
    await audioContext.audioWorklet.addModule('recorder_worklet.js');
    
    const source = audioContext.createMediaStreamSource(mediaStream);
    workletNode = new AudioWorkletNode(audioContext, 'recorder-worklet');
    
    // 3. Initialize resampler
    resampler = new AudioResampler(SAMPLE_RATE, TARGET_RATE);

    // 4. Connect to STT service
    const sttOrigin = new URL(STT_URL).origin;
    sttSocket = io(sttOrigin, {
        path: STT_PATH,
        transports: ['websocket', 'polling'],
        forceNew: true,
        query: { client_id: clientId }
    });

    await new Promise((resolve, reject) => {
        sttSocket.once('connect', resolve);
        sttSocket.once('connect_error', reject);
    });

    // 5. Handle audio from worklet with packetization
    let pending = [];
    let pendingLength = 0;
    const samplesPerPacket = Math.round(SAMPLE_RATE * (PACKET_MS / 1000));

    workletNode.port.onmessage = (event) => {
        const chunk = event.data;
        if (!chunk?.length) return;

        pending.push(chunk);
        pendingLength += chunk.length;

        if (pendingLength >= samplesPerPacket) {
            // Merge pending chunks
            const merged = new Float32Array(pendingLength);
            let offset = 0;
            for (const part of pending) {
                merged.set(part, offset);
                offset += part.length;
            }
            pending = [];
            pendingLength = 0;

            // Resample and send
            const pcm16 = resampler.pushFloat32(merged);
            if (pcm16?.length > 0 && sttSocket?.connected) {
                sttSocket.emit('audio_data', {
                    clientId: clientId,
                    audioData: pcm16.buffer
                });
            }
        }
    };

    // 6. Connect audio graph
    source.connect(workletNode);
    workletNode.connect(audioContext.destination);

    // 7. Subscribe agent server to STT transcripts
    await agentClient.sttSubscribe({
        sttUrl: STT_URL,
        clientId: clientId,
        agent: agentName,
        threadId: threadId
    });
}
```

### 2.5 Stopping STT

```javascript
async function stopSTT(clientId, agentClient) {
    // Disconnect audio
    if (workletNode) {
        workletNode.disconnect();
        workletNode = null;
    }
    if (audioContext) {
        await audioContext.close();
        audioContext = null;
    }
    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
        mediaStream = null;
    }
    if (resampler) {
        resampler.reset();
        resampler = null;
    }

    // Disconnect STT socket
    if (sttSocket) {
        sttSocket.disconnect();
        sttSocket = null;
    }

    // Unsubscribe from agent server
    if (agentClient && clientId) {
        await agentClient.sttUnsubscribe({
            sttUrl: 'https://your-domain.com/stt',
            clientId: clientId
        });
    }
}
```

---

## Part 3: Text-to-Speech (TTS) Integration

### 3.1 TTS State

```javascript
let ttsSocket = null;
let ttsAudioContext = null;
let ttsPlayQueue = Promise.resolve();
```

### 3.2 Enabling TTS

```javascript
async function enableTTS(clientId, agentClient) {
    const TTS_URL = 'https://your-domain.com/tts';
    const TTS_PATH = '/tts/socket.io';

    // 1. Subscribe via agent server
    await agentClient.ttsSubscribe({ clientId });

    // 2. Connect to TTS service as audio sink
    const ttsOrigin = new URL(TTS_URL).origin;
    ttsSocket = io(ttsOrigin, {
        path: TTS_PATH,
        transports: ['websocket', 'polling'],
        forceNew: true,
        query: { 
            type: 'browser', 
            format: 'binary', 
            main_client_id: clientId 
        }
    });

    await new Promise((resolve, reject) => {
        ttsSocket.once('connect', resolve);
        ttsSocket.once('connect_error', reject);
    });

    // 3. Register as audio client
    await new Promise((resolve) => {
        ttsSocket.emit('register_audio_client', {
            main_client_id: clientId,
            connection_type: 'browser',
            mode: 'tts'
        }, resolve);
    });

    // 4. Handle audio chunks
    ttsSocket.on('tts_audio_chunk', async (evt) => {
        const buf = evt?.audio_buffer;
        if (!buf) return;

        const actx = ensureTtsAudioContext();
        let audioBuf;
        try {
            audioBuf = await actx.decodeAudioData(buf.slice(0));
        } catch (e) {
            console.warn('[TTS] decodeAudioData failed:', e);
            return;
        }

        // Queue playback (ensures sequential playback)
        ttsPlayQueue = ttsPlayQueue.then(() => {
            const src = actx.createBufferSource();
            src.buffer = audioBuf;
            src.connect(actx.destination);
            src.start();
            return new Promise(res => { src.onended = res; });
        });
    });

    // 5. Handle stop signal
    ttsSocket.on('tts_stop_immediate', () => {
        closeTtsAudioContext();
    });
}

function ensureTtsAudioContext() {
    if (!ttsAudioContext) {
        ttsAudioContext = new (window.AudioContext || window.webkitAudioContext)({
            sampleRate: 48000
        });
    }
    return ttsAudioContext;
}

async function closeTtsAudioContext() {
    if (ttsAudioContext) {
        try { await ttsAudioContext.close(); } catch {}
        ttsAudioContext = null;
    }
    ttsPlayQueue = Promise.resolve();
}
```

### 3.3 Disabling TTS

```javascript
async function disableTTS(clientId, agentClient) {
    // Unsubscribe via agent server
    if (agentClient && clientId) {
        await agentClient.ttsUnsubscribe({ clientId });
    }

    // Disconnect TTS socket
    if (ttsSocket) {
        ttsSocket.disconnect();
        ttsSocket = null;
    }

    // Close audio context
    await closeTtsAudioContext();
}
```

---

## Part 4: Agent Server Configuration

### 4.1 Agent Preset Files

Agent presets define how the LLM behaves. Create `.agent.json` files in `agent_server/app/data/agents/`.

```json
// example.agent.json
{
    "name": "example",
    "system_prompt": "prompts/example_prompt.txt",
    "params_override": {
        "max_tokens": 512,
        "temperature": 0.7
    },
    "memory_policy": "none",
    "tts_field": null
}
```

### 4.2 Agent Configuration Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Unique agent identifier (lowercase) |
| `system_prompt` | string | Path to system prompt file (relative to agent file) |
| `params_override` | object | LLM parameters to override (temperature, max_tokens, etc.) |
| `memory_policy` | string | `"none"` or `"thread_window"` for conversation memory |
| `tts_field` | string\|null | JSON field to extract for TTS (null = stream all output) |

### 4.3 TTS Field Extraction

When `tts_field` is set, the agent server:
1. Buffers the complete LLM response (doesn't stream to TTS)
2. Parses the response as JSON
3. Extracts the specified field value
4. Sends only that value to TTS

Example for a robot controller agent:

```json
// robot.agent.json
{
    "name": "robot",
    "system_prompt": "prompts/robot_prompt.txt",
    "params_override": {
        "max_tokens": 256,
        "temperature": 0.3
    },
    "memory_policy": "none",
    "tts_field": "Response"
}
```

LLM output:
```json
{ "Operation": "INSPECT", "BoxId": "5", "Response": "Inspecting box 5." }
```

TTS receives only: `"Inspecting box 5."`

### 4.4 Memory Policies

**none** - No conversation history, each request is independent.

**thread_window** - Maintains conversation history per thread_id with a rolling context window.

When using `thread_window`, you must provide a `threadId` when subscribing:

```javascript
await agentClient.sttSubscribe({
    sttUrl: STT_URL,
    clientId: clientId,
    agent: 'my_agent',
    threadId: 'unique-thread-id'  // Required for thread_window
});
```

---

## Part 5: Complete Integration Example

### 5.1 HTML Structure

```html
<!DOCTYPE html>
<html>
<head>
    <title>Voice Assistant</title>
    <script src="https://cdn.socket.io/4.7.2/socket.io.min.js"></script>
</head>
<body>
    <div id="voice-panel">
        <button id="voice-btn">Start Voice</button>
        <div id="status">Click to start</div>
        <div id="transcript"></div>
        <div id="response"></div>
    </div>
    
    <script type="module" src="main.js"></script>
</body>
</html>
```

### 5.2 Main Application

```javascript
// main.js
import { AgentClient } from './agentClient.js';
import { AudioResampler } from './audioResampler.js';

// Configuration
const AGENT_URL = 'https://your-domain.com/llm';
const STT_URL = 'https://your-domain.com/stt';
const TTS_URL = 'https://your-domain.com/tts';
const AGENT_NAME = 'my_agent';

// State
let agentClient = null;
let clientId = null;
let threadId = null;
let voiceActive = false;

// ... (audio state variables from earlier sections)

async function init() {
    // Generate unique IDs
    clientId = crypto.randomUUID();
    threadId = crypto.randomUUID();

    // Connect to agent server
    agentClient = new AgentClient({ url: AGENT_URL });
    await agentClient.connect();

    // Set up transcript handlers
    agentClient.onTranscripts({
        onFinal: (payload) => {
            document.getElementById('transcript').textContent = payload.text;
        }
    });

    // Set up LLM stream handlers
    agentClient.onStream({
        onStarted: () => {
            document.getElementById('response').textContent = '';
        },
        onText: (fullText) => {
            document.getElementById('response').textContent = fullText;
        },
        onDone: () => {
            console.log('Response complete');
        },
        onError: (err) => {
            console.error('Error:', err);
        }
    });

    // Set up voice button
    document.getElementById('voice-btn').onclick = toggleVoice;
}

async function toggleVoice() {
    if (voiceActive) {
        await stopVoice();
    } else {
        await startVoice();
    }
}

async function startVoice() {
    await startSTT(clientId, agentClient, AGENT_NAME, threadId);
    await enableTTS(clientId, agentClient);
    voiceActive = true;
    document.getElementById('status').textContent = 'Listening...';
}

async function stopVoice() {
    await stopSTT(clientId, agentClient);
    await disableTTS(clientId, agentClient);
    voiceActive = false;
    document.getElementById('status').textContent = 'Click to start';
}

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', init);
```

---

## Part 6: Socket.IO Events Reference

### 6.1 Agent Server Events

**Client → Server:**

| Event | Payload | Description |
|-------|---------|-------------|
| `Chat` | `{text, agent, thread_id?, memory?}` | Send text to LLM |
| `Interrupt` | `{}` | Cancel current generation |
| `JoinSTT` | `{sttUrl, clientId, agent, threadId?}` | Subscribe to STT transcripts |
| `LeaveSTT` | `{clientId}` | Unsubscribe from STT |
| `JoinTTS` | `{clientId, voice?, speed?}` | Enable TTS for client |
| `LeaveTTS` | `{clientId}` | Disable TTS for client |

**Server → Client:**

| Event | Payload | Description |
|-------|---------|-------------|
| `RunStarted` | `{runId}` | LLM generation started |
| `ChatChunk` | `{runId, chunk}` | Streamed text chunk |
| `ChatDone` | `{runId}` | Generation complete |
| `Interrupted` | `{runId}` | Generation cancelled |
| `UserTranscript` | `{clientId, text, final, duration}` | STT transcript |
| `STTSubscribed` | `{clientId, sttUrl, agent}` | STT subscription confirmed |
| `TTSSubscribed` | `{clientId}` | TTS subscription confirmed |
| `Error` | `{code, message}` | Error occurred |

### 6.2 STT Service Events

**Client → Server:**

| Event | Payload | Description |
|-------|---------|-------------|
| `audio_data` | `{clientId, audioData}` | Send audio chunk (Int16Array buffer) |

### 6.3 TTS Service Events

**Client → Server:**

| Event | Payload | Description |
|-------|---------|-------------|
| `register_audio_client` | `{main_client_id, connection_type, mode}` | Register as audio sink |

**Server → Client:**

| Event | Payload | Description |
|-------|---------|-------------|
| `tts_audio_chunk` | `{audio_buffer}` | Audio data to play |
| `tts_stop_immediate` | `{}` | Stop playback immediately |

---

## Part 7: Troubleshooting

### Common Issues

**STT not receiving audio:**
- Check microphone permissions
- Verify `audio_data` event payload format: `{clientId, audioData}`
- Ensure audio is resampled to 16kHz

**No transcripts appearing:**
- Verify agent server is subscribed to STT (`JoinSTT` event)
- Check STT service logs for transcription activity
- Ensure `clientId` matches between browser and agent server

**TTS not playing:**
- Check browser autoplay policies (may need user interaction first)
- Verify `tts_audio_chunk` events are being received
- Ensure AudioContext is not suspended

**LLM not responding:**
- Verify agent name exists in agent server
- Check agent server logs for errors
- Ensure memory policy requirements are met (threadId for thread_window)

### Debug Logging

Add these to trace the flow:

```javascript
// Agent client events
agentClient.socket.onAny((event, ...args) => {
    console.log('[Agent]', event, args);
});

// STT socket events
sttSocket.onAny((event, ...args) => {
    console.log('[STT]', event, args);
});

// TTS socket events
ttsSocket.onAny((event, ...args) => {
    console.log('[TTS]', event, args);
});
```

---

## Appendix: File Structure

```
your-app/
├── index.html
├── script/
│   ├── main.js
│   ├── agentClient.js
│   ├── audioResampler.js
│   └── recorder_worklet.js
└── styles/
    └── app.css

agent_server/
└── app/
    ├── main.py
    ├── data/
    │   ├── agents/
    │   │   ├── default.agent.json
    │   │   └── robot.agent.json
    │   └── prompts/
    │       ├── default_prompt.txt
    │       └── robot_prompt.txt
    └── ...
```
