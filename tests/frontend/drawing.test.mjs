// Test the Streamlit component boundary with a small canvas double, not a browser.
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const source = await readFile(new URL("../../app/drawing.js", import.meta.url));
const { default: mount } = await import(
  `data:text/javascript;base64,${source.toString("base64")}`
);

function surface(pixels) {
  let buffer = new Uint8ClampedArray(280 * 280 * 4);
  const calls = [];
  const points = [];
  const context = {
    fillRect() { buffer.fill(0); },
    createImageData() { return { data: new Uint8ClampedArray(buffer.length) }; },
    putImageData(image) { buffer = image.data.slice(); },
    getImageData() { return { data: buffer.slice() }; },
    beginPath() {},
    arc(x, y) { points.push([x, y]); },
    moveTo(x, y) { points.push([x, y]); },
    lineTo(x, y) { points.push([x, y]); },
    fill() {
      const [x, y] = points.at(-1);
      buffer[(Math.floor(y) * 280 + Math.floor(x)) * 4] = 255;
    },
    stroke() { this.fill(); },
  };
  let captured = null;
  const canvas = {
    getContext: () => context,
    // Half-sized display exercises coordinate scaling as well as its offset.
    getBoundingClientRect: () => ({ left: 10, top: 20, width: 140, height: 140 }),
    setPointerCapture: (id) => { captured = id; },
    hasPointerCapture: (id) => captured === id,
    releasePointerCapture: () => { captured = null; },
  };
  const button = {};
  const cleanup = mount({
    parentElement: { querySelector: (name) => name === "canvas" ? canvas : button },
    data: { pixels },
    setStateValue: (key, value) => calls.push({ key, value }),
  });
  const event = (x, y) => ({
    clientX: x, clientY: y, pointerId: 1, preventDefault() {},
  });
  return { canvas, context, button, calls, points, cleanup, event };
}

test("publishes strokes only on release with correctly scaled coordinates", () => {
  const { canvas, calls, points, event } = surface();
  canvas.onpointermove(event(30, 50));
  assert.equal(points.length, 0);
  canvas.onpointerdown(event(30, 50));
  assert.equal(canvas.hasPointerCapture(1), true);
  assert.deepEqual(points[0], [40, 60]);
  canvas.onpointermove(event(40, 60));
  assert.equal(calls.length, 0);
  canvas.onpointerup(event(40, 60));
  assert.equal(canvas.hasPointerCapture(1), false);
  assert.equal(calls.length, 1);
  assert.equal(calls[0].key, "pixels");
  assert.equal(calls[0].value.length, 280 * 280);
  assert.equal(calls[0].value[60 * 280 + 40], 255);
  assert.equal(calls[0].value[80 * 280 + 60], 255);
  canvas.onpointerup(event(40, 60));
  assert.equal(calls.length, 1);
});

test("retains previous ink across remounts and publishes on cancellation", () => {
  const pixels = Array(280 * 280).fill(0);
  pixels[500] = 170;
  const { canvas, calls, event } = surface(pixels);
  assert.equal(calls.length, 0);
  canvas.onpointerdown(event(30, 50));
  canvas.onpointercancel(event(30, 50));
  assert.equal(calls[0].value[500], 170);
  assert.equal(calls[0].value[60 * 280 + 40], 255);
  assert.equal(pixels[60 * 280 + 40], 0);
});

test("clearing resets Python state and cleanup detaches event handlers", () => {
  const { canvas, context, button, calls, cleanup, event } = surface();
  canvas.onpointerdown(event(30, 50));
  canvas.onpointerup(event(30, 50));
  button.onclick();
  assert.deepEqual(calls.at(-1), { key: "pixels", value: null });
  assert.equal(context.getImageData().data.some((pixel) => pixel !== 0), false);
  cleanup();
  for (const name of ["onpointerdown", "onpointermove", "onpointerup", "onpointercancel"]) {
    assert.equal(canvas[name], null);
  }
  assert.equal(button.onclick, null);
});
