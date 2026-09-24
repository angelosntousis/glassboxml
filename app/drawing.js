export default function ({ parentElement, data, setStateValue }) {
  const canvas = parentElement.querySelector("canvas");
  const context = canvas.getContext("2d", { willReadFrequently: true });
  const clear = parentElement.querySelector("button");
  const size = 280;
  let drawing = false;
  const empty = () => { context.fillStyle = "black"; context.fillRect(0, 0, size, size); };
  empty();
  // Restore raw strokes after Python reruns, including optimizer changes.
  if (data?.pixels?.length === size * size) {
    const image = context.createImageData(size, size);
    for (let i = 0; i < data.pixels.length; i++) {
      image.data[i * 4] = image.data[i * 4 + 1] = image.data[i * 4 + 2] = data.pixels[i];
      image.data[i * 4 + 3] = 255;
    }
    context.putImageData(image, 0, 0);
  }
  const point = (event) => {
    const rect = canvas.getBoundingClientRect();
    return [(event.clientX - rect.left) * size / rect.width,
            (event.clientY - rect.top) * size / rect.height];
  };
  const send = () => {
    const rgba = context.getImageData(0, 0, size, size).data;
    const pixels = Array.from({ length: size * size }, (_, i) => rgba[i * 4]);
    setStateValue("pixels", pixels);
  };
  canvas.onpointerdown = (event) => {
    event.preventDefault(); drawing = true;
    canvas.setPointerCapture(event.pointerId);
    context.strokeStyle = "white"; context.fillStyle = "white";
    context.lineWidth = 20; context.lineCap = "round"; context.lineJoin = "round";
    const [x, y] = point(event);
    context.beginPath(); context.arc(x, y, 10, 0, Math.PI * 2); context.fill();
    context.beginPath(); context.moveTo(x, y);
  };
  canvas.onpointermove = (event) => {
    if (!drawing) return;
    event.preventDefault(); const [x, y] = point(event);
    context.lineTo(x, y); context.stroke();
  };
  const finish = (event) => {
    if (!drawing) return;
    drawing = false;
    if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
    send();
  };
  canvas.onpointerup = finish;
  canvas.onpointercancel = finish;
  clear.onclick = () => { drawing = false; empty(); setStateValue("pixels", null); };
  return () => {
    canvas.onpointerdown = canvas.onpointermove = canvas.onpointerup = canvas.onpointercancel = null;
    clear.onclick = null;
  };
}
