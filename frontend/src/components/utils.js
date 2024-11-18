/**
 * Auto crops the image, scales to 28x28 pixel image, and returns as grayscale image.
 * @param {CanvasRenderingContext2D} mainContext - The 2d context of the source canvas.
 * @param {CanvasRenderingContext2D} cropContext - The 2d context of an intermediate hidden canvas.
 * @param {CanvasRenderingContext2D} scaledContext - The 2d context of the destination 28x28 canvas.
 */
import { Chart } from "chart.js";

export function cropScaleGetImageData(mainContext, cropContext, scaledContext) {
  const cropEl = cropContext.canvas;

  // Get the auto-cropped image data and put into the intermediate/hidden canvas
  cropContext.fillStyle = "rgba(255, 255, 255, 255)"; // white non-transparent color
  cropContext.fillRect(0, 0, cropEl.width, cropEl.height);
  cropContext.save();
  const [w, h, croppedImage] = cropImageFromCanvas(mainContext);
  cropEl.width = Math.max(w, h) * 1.2;
  cropEl.height = Math.max(w, h) * 1.2;
  const leftPadding = (cropEl.width - w) / 2;
  const topPadding = (cropEl.height - h) / 2;
  cropContext.putImageData(croppedImage, leftPadding, topPadding);

  // Copy image data to scale 28x28 canvas
  scaledContext.save();
  scaledContext.clearRect(
    0,
    0,
    scaledContext.canvas.height,
    scaledContext.canvas.width
  );
  scaledContext.fillStyle = "rgba(255, 255, 255, 255)"; // white non-transparent color
  scaledContext.fillRect(0, 0, cropEl.width, cropEl.height);
  scaledContext.scale(
    28.0 / cropContext.canvas.width,
    28.0 / cropContext.canvas.height
  );
  scaledContext.drawImage(cropEl, 0, 0);

  // Extract image data and convert into single value (greyscale) array
  const data = rgba2gray(scaledContext.getImageData(0, 0, 28, 28).data);
  scaledContext.restore();

  return data;
}

/**
 * Converts RGBA image data from canvas to grayscale (0 is white & 255 is black).
 * @param {Uint8ClampedArray} data - Image data.
 */
export function rgba2gray(data) {
  let converted = new Float32Array(data.length / 4);

  // Data is stored as [r0,g0,b0,a0, ... r[n],g[n],b[n],a[n]] where n is number of pixels.
  for (let i = 0; i < data.length; i += 4) {
    let r = 255 - data[i]; // red
    let g = 255 - data[i + 1]; // green
    let b = 255 - data[i + 2]; // blue
    // Alpha channel is ignored as it's assumed to be 255

    // Use RGB grayscale coefficients
    let y = 0.299 * r + 0.587 * g + 0.114 * b;
    converted[i / 4] = y; // 4 times fewer data points but the same number of pixels.
  }
  return converted;
}

/**
 * Auto crops a canvas images and returns its image data.
 * @param {CanvasRenderingContext2D} ctx - Canvas 2D context.
 */
export function cropImageFromCanvas(ctx) {
  let canvas = ctx.canvas,
    w = canvas.width,
    h = canvas.height,
    pix = { x: [], y: [] },
    imageData = ctx.getImageData(0, 0, canvas.width, canvas.height),
    x,
    y,
    index;

  for (y = 0; y < h; y++) {
    for (x = 0; x < w; x++) {
      index = (y * w + x) * 4;

      let r = imageData.data[index];
      let g = imageData.data[index + 1];
      let b = imageData.data[index + 2];
      if (Math.min(r, g, b) !== 255) {
        pix.x.push(x);
        pix.y.push(y);
      }
    }
  }
  pix.x.sort((a, b) => a - b);
  pix.y.sort((a, b) => a - b);
  let n = pix.x.length - 1;

  // No drawing detected
  if (n === -1) {
    return [0, 0, ctx.getImageData(0, 0, 1, 1)];
  }

  w = 1 + pix.x[n] - pix.x[0];
  h = 1 + pix.y[n] - pix.y[0];
  return [w, h, ctx.getImageData(pix.x[0], pix.y[0], w, h)];
}

/**
 * Truncates number to a given decimal position
 * @param {number} num - Number to truncate.
 * @param {number} fixed - Decimal positions.
 */
export function toFixed(num, fixed) {
  const re = new RegExp("^-?\\d+(?:\\.\\d{0," + (fixed || -1) + "})?");
  return num.toString().match(re)[0];
}

/**
 * Helper function that builds a chart using Chart.js library.
 * @param {HTMLCanvasElement} chartEl - Chart canvas element.
 */
export function chartConfigBuilder(chartEl) {
  return new Chart(chartEl, {
    type: "bar",
    data: {
      labels: ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
      datasets: [
        {
          data: Array(10).fill(0.0),
          borderWidth: 0,
          fill: true,
          backgroundColor: "#247ABF",
        },
      ],
    },
    options: {
      responsive: false,
      maintainAspectRatio: false,
      animation: true,
      plugins: {
        legend: {
          display: false,
        },
        tooltip: {
          enabled: true,
        },
        datalabels: {
          color: "white",
          formatter: (value) => toFixed(value, 2),
        },
      },
      scales: {
        y: {
          beginAtZero: true,
          max: 1.0,
        },
      },
    },
  });
}
