# Quickstart: Text-only Chat

Copy-paste:

```html
<script src="./socket.io.min.js"></script>
<script type="module">
	import { AgentClient } from "../agentClient.js";

	const client = new AgentClient({ url: location.origin });
	await client.connect();
	client.onStream({ onText: t => console.log(t) });

	const threadId = crypto.randomUUID();
	await client.runText("Hello!", { agent: "topic", threadId });
</script>
```
