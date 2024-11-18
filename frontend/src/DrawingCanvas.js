import React, { useRef, useEffect } from "react";
import { Canvas as FabricCanvas } from "fabric";
import { cropScaleGetImageData } from "./utils";

const DrawingCanvas = React.forwardRef(({ onDraw }, ref) => {
  const fabricCanvasRef = useRef(null);

  useEffect(() => {
    const fabricCanvas = new FabricCanvas(ref.current, {
      isDrawingMode: true,
    });
    fabricCanvas.freeDrawingBrush.width = 25;
    fabricCanvas.backgroundColor = "rgba(255, 255, 255, 255)";
    fabricCanvasRef.current = fabricCanvas;

    fabricCanvas.on("mouse:up", () => onDraw());

    return () => fabricCanvas.dispose();
  }, [onDraw, ref]);

  return (
    <canvas
      id="main-canvas"
      ref={ref}
      width="300"
      height="300"
      style={{ border: "1px solid #aaa" }}
    />
  );
});

export default DrawingCanvas;
