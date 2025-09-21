class RecorderWorklet extends AudioWorkletProcessor {
	process(inputs) {
		const input = inputs[0];
		// mono channel 0
		if (input && input[0] && input[0].length) {
			// send a copy to the main thread
			this.port.postMessage(input[0].slice());
		}
		return true; // keep node alive
	}
}
registerProcessor('recorder-worklet', RecorderWorklet);
