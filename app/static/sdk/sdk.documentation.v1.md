# AgentClient (Browser SDK) — Quick Docs

A tiny ES6 client for talking to your `agent_server` over Socket.IO. It handles:

* Connecting/reconnecting (with backoff)
* Starting a run and streaming text as it’s generated
* Aggregating chunks into full text for you (no “one word per line” surprises)
* Canceling an in-flight run

> Memory is **server-controlled**. Clients just pick an **agent** (e.g. `"topic"` or `"router"`) and optionally provide a `threadId` when the agent uses memory.

---

## Install / Include

Load Socket.IO (global `io`) and import the SDK:

```html
<script src="/socket.io.min.js"></script>
<script type="module">
	import { AgentClient } from "/sdk/agentClient.js";
	// ...
</script>
```

* If your static paths differ, adjust the URLs accordingly.
* This SDK is intended for **browser** usage.

---

## Quick Start

```html
<script type="module">
	import { AgentClient } from "/sdk/agentClient.js";

	const client = new AgentClient({ url: location.origin });

	await client.connect({
		onReconnect: (attempt) => console.log("[reconnected]", attempt),
	});

const threadId = crypto.randomUUID(); // keep this stable for the same conversation

	const result = await client.runText(
		"Hello there!",
		{ agent: "topic", threadId },   // agent name required; threadId optional
		{
			onStarted: (runId) => console.log("[run]", runId),
			onText:    (full)  => console.log("[text]", full),   // aggregated text so far
			onDone:    ()      => console.log("[done]"),
			onError:   (err)   => console.error("[error]", err),
		}
	);

	console.log("Final:", result.runId, result.text);
</script>
```

---

## API Reference

### `new AgentClient(opts?)`

```ts
type AgentClientOpts = {
	url?: string;   // default: window.location.origin
	path?: string;  // default: "/socket.io"
};
```

Creates a new client instance. One instance holds a single Socket.IO connection.

---

### `await client.connect(options?)`

```ts
type ConnectOptions = {
	onReconnect?: (attempt: number) => void; // called after a successful reconnect
};
```

* Establishes the socket and installs stream handlers.
* Uses Socket.IO’s built-in exponential backoff.
* Resolves after the **first** successful connect.

> If the socket drops mid-run, the in-flight run will reject with `INTERRUPTED`. You can retry or update UI in `onReconnect`.

---

### `await client.runText(text, options, callbacks?)`

```ts
type RunOptions = {
	agent: string;         // e.g. "topic" or "router" (must match a server preset)
	threadId?: string;     // recommended if the agent uses memory
};

type RunCallbacks = {
	onStarted?: (runId: string) => void;                   // when server starts the run
	onChunk?:   (delta: string) => void;                   // raw partial deltas
	onText?:    (full: string)  => void;                   // aggregated text so far (preferred)
	onDone?:    () => void;                                // stream finished cleanly
	onError?:   (err: { code: string, message: string }) => void; // server-side error/interruption
};
```

Returns a Promise that resolves to:

```ts
{ runId: string | null, text: string } // final aggregated text
```

**Notes**

* Only **one active run** per `AgentClient` at a time. Starting another before the last finishes throws an error.
* `onText` is the easy way to render a streaming transcript, no per-word line breaks.

---

### `client.cancel(runId?) : boolean`

Cancels the current run. If `runId` is provided, it must match the active run; otherwise it cancels whatever is active. Returns `true` if an Interrupt was sent.

```js
const ok = client.cancel();
if (!ok) console.log("No active run to cancel");
```

The pending `runText()` promise rejects with an `INTERRUPTED` error.

---

## Usage Patterns

### Render to the DOM

```js
const outEl = document.getElementById("out");

await client.runText(
	promptEl.value.trim(),
	{ agent: "topic", threadId },
	{
		onStarted: (id) => console.log("[run]", id),
		onText: (full) => { outEl.textContent = full; },
		onDone: () => console.log("[done]"),
		onError: (e) => console.error("[error]", e),
	}
);
```

### Cancel a Run

```js
document.getElementById("cancelBtn").addEventListener("click", () => {
	const ok = client.cancel();
	console.log(ok ? "Cancel sent" : "No active run");
});
```

### Reconnect Handling

```js
await client.connect({
	onReconnect: (attempt) => {
		showBanner(`Reconnected (attempt ${attempt})`);
		// Optionally restart a run or just allow the user to continue.
	}
});
```

---

## Memory & Threads (Concepts)

* **Server controls memory policy.** Clients do **not** choose memory modes.
* If an agent uses memory (e.g., “thread window”), provide a **stable `threadId`** across turns to keep context.

  * Use `crypto.randomUUID()` to generate one per conversation.
* If an agent is stateless, you can omit `threadId`.

---

## Error Model

* `runText()` rejects with:

  * `{ code: "INTERRUPTED" }` if you cancel or the socket drops mid-run.
  * `{ code: "...", message: "..." }` for server-reported errors (e.g., bad agent name).
* The SDK also forwards errors to `onError`.

---

## FAQ

**Why two callbacks (`onChunk` and `onText`)?**
`onChunk` gives you raw deltas (tokenish pieces). `onText` gives you the **aggregated** text so you can just assign it to a DOM node without managing whitespace or joins.

**Can I run multiple prompts concurrently?**
Use **one `AgentClient` instance per concurrent run** (or sequence your calls). The server enforces one active run per socket.

**Do I need to send a session ID?**
No. If your chosen agent uses memory, supply a `threadId`. Otherwise it’s optional.

---

## Minimal Example Page

```html
<!doctype html>
<html>
<head>
	<meta charset="utf-8"/>
	<title>AgentClient demo</title>
	<script src="/socket.io.min.js"></script>
</head>
<body>
	<input id="prompt" placeholder="Say hi…" />
	<button id="send">Send</button>
	<button id="cancel">Cancel</button>
	<pre id="out"></pre>
	<pre id="log"></pre>

	<script type="module">
		import { AgentClient } from "/sdk/agentClient.js";
		const client = new AgentClient({ url: location.origin });

		const out = document.getElementById("out");
		const log = (...a) => (document.getElementById("log").textContent += a.join(" ") + "\n");

		const threadId = crypto.randomUUID();
		await client.connect({ onReconnect: (attempt) => log("[reconnected]", attempt) });

		document.getElementById("send").onclick = async () => {
			out.textContent = "";
			try {
				await client.runText(
					document.getElementById("prompt").value,
					{ agent: "topic", threadId },
					{
						onStarted: (id) => log("[run]", id),
						onText: (full) => { out.textContent = full; },
						onDone: () => log("[done]"),
						onError: (e) => log("[error]", e.message || e.code),
					}
				);
			} catch (e) {
				log("[failed]", e.message || e.code);
			}
		};

		document.getElementById("cancel").onclick = () => {
			log(client.cancel() ? "[cancel sent]" : "[no active run]");
		};
	</script>
</body>
</html>
```

---

## Versioning & Compatibility

* Protocol version: **1.0.0** (server may include this in a hello/ack).
* Event names expected by the SDK:

  * `RunStarted` → `{ runId }`
  * `ChatChunk` → `{ chunk }`
  * `ChatDone`  → `{}`
  * `Interrupted` → `{}`
  * `Error` → `{ code, message, runId? }`

If your server changes these, bump the SDK and update this doc.

---

## Tips

* Keep an `AgentClient` per browser tab/page. If you need parallel runs, create multiple clients.
* Always generate and reuse a `threadId` for conversations that should carry context.
* Use `onText` to keep your UI in sync without worrying about token boundaries.

---

Questions or improvements you’d like in the SDK? Add them and we can iterate.
