import React, { useEffect, useRef } from "react";
import { fabric } from "fabric";
import { cropScaleGetImageData } from "./utils";
import { default as wasm, Mnist } from "../pkg/mnist_inference_web.js";

export const FabricCanvas = ({ onInference, clearCanvas, setClearCanvas }) => {
  const canvasRef = useRef(null);
  const mainContextRef = useRef(null);
  const cropCanvasRef = useRef(null);
  const scaledCanvasRef = useRef(null);
  const mnistRef = useRef(null);
  let timeoutId = useRef(null);
  let isDrawing = useRef(false);
  let isTimeOutSet = useRef(false);

  useEffect(() => {
    const initWasm = async () => {
      await wasm();
      mnistRef.current = new Mnist();
    };

    initWasm();

    const canvasEl = canvasRef.current;
    const mainContext = canvasEl.getContext("2d", { willReadFrequently: true });
    mainContextRef.current = mainContext;

    const cropCanvasEl = document.createElement("canvas");
    cropCanvasEl.width = 28;
    cropCanvasEl.height = 28;
    const cropContext = cropCanvasEl.getContext("2d", {
      willReadFrequently: true,
    });
    cropCanvasRef.current = cropContext;

    const scaledCanvasEl = document.createElement("canvas");
    scaledCanvasEl.width = 28;
    scaledCanvasEl.height = 28;
    const scaledContext = scaledCanvasEl.getContext("2d", {
      willReadFrequently: true,
    });
    scaledCanvasRef.current = scaledContext;

    const fabricCanvas = new fabric.Canvas(canvasEl, {
      isDrawingMode: true,
    });

    fabricCanvas.freeDrawingBrush.width = 25;
    fabricCanvas.backgroundColor = "rgba(255, 255, 255, 255)";
    fabricCanvas.renderAll();

    const fireOffInference = async () => {
      clearTimeout(timeoutId.current);
      timeoutId.current = setTimeout(async () => {
        isTimeOutSet.current = true;
        fabricCanvas.freeDrawingBrush._finalizeAndAddPath();
        const data = cropScaleGetImageData(
          mainContextRef.current,
          cropCanvasRef.current,
          scaledCanvasRef.current
        );
        const output = await mnistRef.current.inference(data);
        onInference(output);
        isTimeOutSet.current = false;
      }, 50);
      isTimeOutSet.current = true;
    };

    fabricCanvas.on("mouse:down", () => {
      isDrawing.current = true;
    });

    fabricCanvas.on("mouse:up", () => {
      isDrawing.current = false;
      fireOffInference();
    });

    fabricCanvas.on("mouse:move", () => {
      if (isDrawing.current && !isTimeOutSet.current) {
        fireOffInference();
      }
    });

    if (clearCanvas) {
      fabricCanvas.clear();
      fabricCanvas.backgroundColor = "rgba(255, 255, 255, 255)";
      fabricCanvas.renderAll();
      mainContext.clearRect(0, 0, canvasEl.width, canvasEl.height);
      setClearCanvas(false);
    }

    // Cleanup on unmount
    return () => {
      fabricCanvas.dispose();
    };
  }, [onInference, clearCanvas, setClearCanvas]);

  return <canvas id="main-canvas" ref={canvasRef} width="300" height="300" />;
};
