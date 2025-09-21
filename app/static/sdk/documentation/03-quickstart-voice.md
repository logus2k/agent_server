# Quickstart: Voice + Server-side STT

```html
<script src="./socket.io.min.js"></script>
<script type="module">
	import { AgentClient } from "../agentClient.js";
	import { AudioResampler } from "../audioResampler.js";

	const client = new AgentClient({ url: location.origin });
	await client.connect();
	client.onStream({ onText: t => console.log(t) });

	const threadId = crypto.randomUUID();
	await client.sttSubscribe({
		sttUrl: "http://localhost:2700",
		clientId: "demo",
		agent: "topic",
		threadId
	});

	const ctx = new AudioContext({ sampleRate: 48000, latencyHint: "interactive" });
	await ctx.audioWorklet.addModule("../recorder.worklet.js");
	const src = ctx.createMediaStreamSource(await navigator.mediaDevices.getUserMedia({ audio: true }));
	const worklet = new AudioWorkletNode(ctx, "recorder-worklet");
	const resampler = new AudioResampler(ctx.sampleRate, 16000);

	const stt = io("http://localhost:2700", { transports:["websocket"] });
	worklet.port.onmessage = e => {
		const pcm16 = resampler.pushFloat32(e.data);
		if (pcm16) stt.emit("audio_data", { clientId: "demo", audioData: pcm16.buffer });
	};
	src.connect(worklet);
</script>
```
