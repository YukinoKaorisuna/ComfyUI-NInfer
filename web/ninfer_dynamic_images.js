// NInfer Local LLM — progressive image sockets (rgthree-style stabilization).
//
// Starts with "image" + "images". Wire "images" and a free "image3" appears; wire
// "image3" and "image4" appears; up to "image8". Unwired sockets beyond the wired
// run are removed, so the node never shows a pile of dangling inputs.
//
// Implementation notes (learned the hard way):
//   - addInput takes POSITIONAL (name, type) strings — passing an object renders
//     the socket label as "[object Object]".
//   - onConnectionsChange side arg is NOT reliably the number 1 across frontends,
//     so we re-sync on any connection change (debounced, to avoid re-entry while
//     we ourselves add/remove sockets).
//   - The backend declares all eight optional inputs, so anything we add is valid
//     at execution time and saved workflows keep working.
import { app } from "../../scripts/app.js";

const NODE_NAME = "NinferLocalLLM";
const DYN_SOCKETS = ["image3", "image4", "image5", "image6", "image7", "image8"];

const pending = new WeakMap();

function syncImageSockets(node) {
    let highestWired = -1;
    let imagesWired = false;
    for (const input of node.inputs || []) {
        const idx = DYN_SOCKETS.indexOf(input.name);
        if (idx >= 0) {
            if (input.link != null) highestWired = Math.max(highestWired, idx);
        } else if (input.name === "images") {
            imagesWired = imagesWired || input.link != null;
        }
    }
    // Socket budget: every wired dynamic socket keeps its place, and exactly one
    // free socket trails the wired run. With nothing dynamic wired, "images"
    // being wired is what summons the first free slot (image3).
    let count = highestWired >= 0 ? highestWired + 2 : imagesWired ? 1 : 0;
    count = Math.min(count, DYN_SOCKETS.length);

    let changed = false;
    for (let k = 0; k < count; k++) {
        const name = DYN_SOCKETS[k];
        if (!node.inputs.some((input) => input.name === name)) {
            node.addInput(name, "IMAGE");
            changed = true;
        }
    }
    for (const input of (node.inputs || []).slice()) {
        const idx = DYN_SOCKETS.indexOf(input.name);
        if (idx >= 0 && idx >= count && input.link == null) {
            node.removeInput(node.inputs.indexOf(input));
            changed = true;
        }
    }
    if (changed) {
        if (node.computeSize) node.setSize(node.computeSize());
        if (app.graph) app.graph.setDirtyCanvas(true, true);
    }
}

function scheduleSync(node) {
    if (pending.has(node)) return;
    pending.set(
        node,
        setTimeout(() => {
            pending.delete(node);
            syncImageSockets(node);
        }, 50),
    );
}

app.registerExtension({
    name: "NInfer.DynamicImageInputs",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated
                ? onNodeCreated.apply(this, arguments)
                : undefined;
            syncImageSockets(this);
            return result;
        };

        // Deliberately no side/type filtering here: frontends disagree on what the
        // first argument is (number vs string), and re-syncing on output-side
        // changes is harmless.
        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function () {
            const result = onConnectionsChange
                ? onConnectionsChange.apply(this, arguments)
                : undefined;
            scheduleSync(this);
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            syncImageSockets(this);
            return result;
        };
    },
});
